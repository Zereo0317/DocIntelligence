This is others approach on building knowledge graph:
"""Entity-Relationship extraction module."""
import asyncio
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

from fast_graphrag._llm import BaseLLMService, format_and_send_prompt
from fast_graphrag._models import TQueryEntities
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._storage._gdb_igraph import IGraphStorage, IGraphStorageConfig
from fast_graphrag._types import GTId, TChunk, TEntity, TGraph, TRelation
from fast_graphrag._utils import logger

from ._base import BaseInformationExtractionService


class TGleaningStatus(BaseModel):
    status: Literal["done", "continue"] = Field(
        description="done if all entities and relationship have been extracted, continue otherwise"
    )


@dataclass
class DefaultInformationExtractionService(BaseInformationExtractionService[TChunk, TEntity, TRelation, GTId]):
    """Default entity and relationship extractor."""

    def extract(
        self,
        llm: BaseLLMService,
        documents: Iterable[Iterable[TChunk]],
        prompt_kwargs: Dict[str, str],
        entity_types: List[str],
    ) -> List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, GTId]]]]:
        """Extract both entities and relationships from the given data."""
        return [
            asyncio.create_task(self._extract(llm, document, prompt_kwargs, entity_types)) for document in documents
        ]

    async def extract_entities_from_query(
        self, llm: BaseLLMService, query: str, prompt_kwargs: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Extract entities from the given query."""
        prompt_kwargs["query"] = query
        entities, _ = await format_and_send_prompt(
            prompt_key="entity_extraction_query",
            llm=llm,
            format_kwargs=prompt_kwargs,
            response_model=TQueryEntities,
        )

        return {
            "named": entities.named,
            "generic": entities.generic
        }

    async def _extract(
        self, llm: BaseLLMService, chunks: Iterable[TChunk], prompt_kwargs: Dict[str, str], entity_types: List[str]
    ) -> Optional[BaseGraphStorage[TEntity, TRelation, GTId]]:
        """Extract both entities and relationships from the given chunks."""
        # Extract entities and relatioships from each chunk
        try:
            chunk_graphs = await asyncio.gather(
                *[self._extract_from_chunk(llm, chunk, prompt_kwargs, entity_types) for chunk in chunks]
            )
            if len(chunk_graphs) == 0:
                return None

            # Combine chunk graphs in document graph
            return await self._merge(llm, chunk_graphs)
        except Exception as e:
            logger.error(f"Error during information extraction from document: {e}")
            return None

    async def _gleaning(
        self, llm: BaseLLMService, initial_graph: TGraph, history: list[dict[str, str]]
    ) -> Optional[TGraph]:
        """Do gleaning steps until the llm says we are done or we reach the max gleaning steps."""
        # Prompts
        current_graph = initial_graph

        try:
            for gleaning_count in range(self.max_gleaning_steps):
                # Do gleaning step
                gleaning_result, history = await format_and_send_prompt(
                    prompt_key="entity_relationship_continue_extraction",
                    llm=llm,
                    format_kwargs={},
                    response_model=TGraph,
                    history_messages=history,
                )

                # Combine new entities, relationships with previously obtained ones
                current_graph.entities.extend(gleaning_result.entities)
                current_graph.relationships.extend(gleaning_result.relationships)

                # Stop gleaning if we don't need to keep going
                if gleaning_count == self.max_gleaning_steps - 1:
                    break

                # Ask llm if we are done extracting entities and relationships
                gleaning_status, _ = await format_and_send_prompt(
                    prompt_key="entity_relationship_gleaning_done_extraction",
                    llm=llm,
                    format_kwargs={},
                    response_model=TGleaningStatus,
                    history_messages=history,
                )

                # If we are done parsing, stop gleaning
                if gleaning_status.status == Literal["done"]:
                    break
        except Exception as e:
            logger.error(f"Error during gleaning: {e}")

            return None

        return current_graph

    async def _extract_from_chunk(
        self, llm: BaseLLMService, chunk: TChunk, prompt_kwargs: Dict[str, str], entity_types: List[str]
    ) -> TGraph:
        """Extract entities and relationships from the given chunk."""
        prompt_kwargs["input_text"] = chunk.content

        chunk_graph, history = await format_and_send_prompt(
            prompt_key="entity_relationship_extraction",
            llm=llm,
            format_kwargs=prompt_kwargs,
            response_model=TGraph,
        )

        # Do gleaning
        chunk_graph_with_gleaning = await self._gleaning(llm, chunk_graph, history)
        if chunk_graph_with_gleaning:
            chunk_graph = chunk_graph_with_gleaning

        _clean_entity_types = [re.sub("[ _]", "", entity_type).upper() for entity_type in entity_types]
        for entity in chunk_graph.entities:
            if re.sub("[ _]", "", entity.type).upper() not in _clean_entity_types:
                entity.type = "UNKNOWN"

        # Assign chunk ids to relationships
        for relationship in chunk_graph.relationships:
            relationship.chunks = [chunk.id]

        return chunk_graph

    async def _merge(self, llm: BaseLLMService, graphs: List[TGraph]) -> BaseGraphStorage[TEntity, TRelation, GTId]:
        """Merge the given graphs into a single graph storage."""
        graph_storage = IGraphStorage[TEntity, TRelation, GTId](config=IGraphStorageConfig(TEntity, TRelation))

        await graph_storage.insert_start()

        try:
            # This is synchronous since each sub graph is inserted into the graph storage and conflicts are resolved
            for graph in graphs:
                await self.graph_upsert(llm, graph_storage, graph.entities, graph.relationships)
        finally:
            await graph_storage.insert_done()

        return graph_storage


"""LLM Services module."""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from pydantic import BaseModel

from fast_graphrag._models import BaseModelAlias
from fast_graphrag._prompt import PROMPTS

T_model = TypeVar("T_model", bound=Union[BaseModel, BaseModelAlias])


async def format_and_send_prompt(
    prompt_key: str,
    llm: "BaseLLMService",
    format_kwargs: dict[str, Any],
    response_model: Type[T_model],
    **args: Any,
) -> Tuple[T_model, list[dict[str, str]]]:
    """Get a prompt, format it with the supplied args, and send it to the LLM.

    Args:
        prompt_key (str): The key for the prompt in the PROMPTS dictionary.
        llm (BaseLLMService): The LLM service to use for sending the message.
        response_model (Type[T_model]): The expected response model.
        format_kwargs (dict[str, Any]): Dictionary of arguments to format the prompt.
        model (str | None): The model to use for the LLM. Defaults to None.
        max_tokens (int | None): The maximum number of tokens for the response. Defaults to None.
        **args (Any): Additional keyword arguments to pass to the LLM.

    Returns:
        T_model: The response from the LLM.
    """
    # Get the prompt from the PROMPTS dictionary
    prompt = PROMPTS[prompt_key]

    # Format the prompt with the supplied arguments
    formatted_prompt = prompt.format(**format_kwargs)

    # Send the formatted prompt to the LLM
    return await llm.send_message(prompt=formatted_prompt, response_model=response_model, **args)


@dataclass
class BaseLLMService:
    """Base class for Language Model implementations."""

    model: Optional[str] = field(default=None)
    base_url: Optional[str] = field(default=None)
    api_key: Optional[str] = field(default=None)
    llm_async_client: Any = field(init=False, default=None)

    async def send_message(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model: Type[T_model] | None = None,
        **kwargs: Any,
    ) -> Tuple[T_model, list[dict[str, str]]]:
        """Send a message to the language model and receive a response.

        Args:
            prompt (str): The input message to send to the language model.
            model (str): The name of the model to use.
            system_prompt (str, optional): The system prompt to set the context for the conversation. Defaults to None.
            history_messages (list, optional): A list of previous messages in the conversation. Defaults to empty.
            response_model (Type[T], optional): The Pydantic model to parse the response. Defaults to None.
            **kwargs: Additional keyword arguments that may be required by specific LLM implementations.

        Returns:
            str: The response from the language model.
        """
        raise NotImplementedError


@dataclass
class BaseEmbeddingService:
    """Base class for Language Model implementations."""

    embedding_dim: int = field(default=1536)
    model: Optional[str] = field(default="text-embedding-3-small")
    base_url: Optional[str] = field(default=None)
    api_key: Optional[str] = field(default=None)

    embedding_async_client: Any = field(init=False, default=None)

    async def encode(
        self, texts: list[str], model: Optional[str] = None
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Get the embedding representation of the input text.

        Args:
            texts (str): The input text to embed.
            model (str): The name of the model to use.

        Returns:
            list[float]: The embedding vector as a list of floats.
        """
        raise NotImplementedError


This is how others design their prompt. You should make it suitable for generating concept behavior graph
"""Prompts."""

from typing import Any, Dict

PROMPTS: Dict[str, Any] = {}

## NEW
PROMPTS["entity_relationship_extraction"] = """# DOMAIN PROMPT
{domain}

# GOAL
Your goal is to highlight information that is relevant to the domain and the questions that may be asked on it.
Given an input document, identify all relevant entities and all relationships among them.

Examples of possible questions:
{example_queries}

# STEPS
1. Identify all entities of the given types. Make sure to extract all and only the entities that are of one of the given types. Use singular names and split compound concepts when necessary (for example, from the sentence "they are movie and theater directors", you should extract the entities "movie director" and "theater director").
2. Identify all relationships between the entities found in step 1. Clearly resolve pronouns to their specific names to maintain clarity.
3. Double check that each entity identified in step 1 appears in at least one relationship. If not, add the missing relationships.

# EXAMPLE DATA
Example types: [location, organization, person, communication]
Example document: Radio City: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into new media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."

Output:
{{
"entities": [
	{{"name": "RADIO CITY", "type": "organization", "desc": "Radio City is India's first private FM radio station"}},
	{{"name": "INDIA", "type": "location", "desc": "A country"}},
	{{"name": "FM RADIO STATION", "type": "communication", "desc": "A radio station that broadcasts using frequency modulation"}},
	{{"name": "ENGLISH", "type": "communication", "desc": "A language"}},
	{{"name": "HINDI", "type": "communication", "desc": "A language"}},
	{{"name": "NEW MEDIA", "type": "communication", "desc": "New media"}},
	{{"name": "PLANETRADIOCITY", "type": "organization", "desc": "PlanetRadiocity.com is an online music portal"}},
	{{"name": "MUSIC PORTAL", "type": "communication", "desc": "A website that offers music related information"}},
	{{"name": "NEWS", "type": "communication", "desc": "News"}},
	{{"name": "VIDEO", "type": "communication", "desc": "Video"}},
	{{"name": "SONG", "type": "communication", "desc": "Song"}}
],
"relationships": [
	{{"source": "RADIO CITY", "target": "INDIA", "desc": "Radio City is located in India"}},
	{{"source": "RADIO CITY", "target": "FM RADIO STATION", "desc": "Radio City is a private FM radio station started on 3 July 2001"}},
	{{"source": "RADIO CITY", "target": "ENGLISH", "desc": "Radio City broadcasts English songs"}},
	{{"source": "RADIO CITY", "target": "HINDI", "desc": "Radio City broadcasts songs in the Hindi language"}},
	{{"source": "RADIO CITY", "target": "PLANETRADIOCITY", "desc": "Radio City launched PlanetRadiocity.com in May 2008"}},
	{{"source": "PLANETRADIOCITY", "target": "MUSIC PORTAL", "desc": "PlanetRadiocity.com is a music portal"}},
	{{"source": "PLANETRADIOCITY", "target": "NEWS", "desc": "PlanetRadiocity.com offers music related news"}},
	{{"source": "PLANETRADIOCITY", "target": "SONG", "desc": "PlanetRadiocity.com offers songs"}}
],
"other_relationships": [
	{{"source": "RADIO CITY", "target": "NEW MEDIA", "desc": "Radio City forayed into new media in May 2008."}},
	{{"source": "PLANETRADIOCITY", "target": "VIDEO", "desc": "PlanetRadiocity.com offers music related videos"}}
]
}}

# INPUT DATA
Types: {entity_types}
Document: {input_text}

Output:
"""

PROMPTS["entity_relationship_continue_extraction"] = "MANY entities were missed in the last extraction.  Add them below using the same format:"

PROMPTS["entity_relationship_gleaning_done_extraction"] = "Retrospectively check if all entities have been correctly identified: answer done if so, or continue if there are still entities that need to be added."

PROMPTS["entity_extraction_query"] = """Given the query below, your task is to extract all entities relevant to perform information retrieval to produce an answer.

-EXAMPLE 1-
Query: Who directed the film that was shot in or around Leland, North Carolina in 1986?
Ouput: {{"named": ["[PLACE] Leland", "[COUNTRY] North Carolina", "[YEAR] 1986"], "generic": ["film director"]}}

-EXAMPLE 2-
Query: What relationship does Fred Gehrke have to the 23rd overall pick in the 2010 Major League Baseball Draft?
Ouput: {{"named": ["[BASEBALL PLAYER] Fred Gehrke", "[EVENT] 2010 Major League Baseball Draft"], "generic": ["23rd baseball draft pick"]}}

-INPUT-
Query: {query}
Output:
"""


PROMPTS[
	"summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a summary of the data provided below.
Given the current description, summarize it by removing redundant and generic information. Resolve any contradictions and provide a single, coherent summary.
Write in third person and explicitly include the entity names to preserve the full context.

Current:
{description}

Updated:
"""


PROMPTS[
	"edges_group_similar"
] = """You are a helpful assistant responsible for maintaining a list of facts describing the relations between two entities so that information is not redundant.
Given a list of ids and facts, identify any facts that should be grouped together as they contain similar or duplicated information and provide a new summarized description for the group.

# EXAMPLE
Facts (id, description):
0, Mark is the dad of Luke
1, Luke loves Mark
2, Mark is always ready to help Luke
3, Mark is the father of Luke
4, Mark loves Luke very much

Ouput:
{{
	grouped_facts: [
	{{
		'ids': [0, 3],
		'description': 'Mark is the father of Luke'
	}},
	{{
		'ids': [1, 4],
		'description': 'Mark and Luke love each other very much'
	}}
	]
}}

# INPUT:
Facts:
{edge_list}

Ouput:
"""

PROMPTS["generate_response_query_with_references"] = """You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a response to the user query using the relevant information in the input data:
- the "Entities" and "Relationships" tables contain high-level information. Use these tables to identify the most important entities and relationships to respond to the query.
- the "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

Follow these steps:
1. Read and understand the user query.
2. Look at the "Entities" and "Relationships" tables to get a general sense of the data and understand which information is the most relevant to answer the query.
3. Carefully analyze all the "Sources" to get more detailed information. Information could be scattered across several sources, use the identified relevant entities and relationships to guide yourself through the analysis of the sources.
4. While you write the response, you must include inline references to the all the sources you are using by appending `[<source_id>]` at the end of each sentence, where `source_id` is the corresponding source ID from the "Sources" list.
5. Write the response to the user query - which must include the inline references - based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.

Answer:
"""

PROMPTS["generate_response_query_no_references"] = """You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a response to the user query using the relevant information in the input data:
- the "Entities" and "Relationships" tables contain high-level information. Use these tables to identify the most important entities and relationships to respond to the query.
- the "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

Follow these steps:
1. Read and understand the user query.
2. Look at the "Entities" and "Relationships" tables to get a general sense of the data and understand which information is the most relevant to answer the query.
3. Carefully analyze all the "Sources" to get more detailed information. Information could be scattered across several sources, use the identified relevant entities and relationships to guide yourself through the analysis of the sources.
4. Write the response to the user query based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.

Answer:
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
