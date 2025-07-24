import logging
from typing import Optional, Any
from dataclasses import dataclass
import json

import dspy
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.genie import Genie, GenieResponse
from databricks_dspy.clients.databricks_lm import DatabricksLM

logger = logging.getLogger(__name__)

# TODO: unit tests

@dataclass
class GenieToolResponse:
    text: Optional[str]
    conversation_id: Optional[str]
    query: Optional[str] = None
    data: Optional[list[dict[str, Any]]] = None

    def to_dict(self):
        return {
            "text": self.text,
            "conversation_id": self.conversation_id,
            "query": self.query,
            "data": self.data,
        }

    @staticmethod
    @mlflow.trace(span_type="PARSER")
    def from_genie_response(genie_response: GenieResponse):
        text_response = (
                str(genie_response.result)
                if genie_response.description is None
                else genie_response.description
            )
        try:
            parsed_data = (
                json.loads(str(genie_response.result))
                if genie_response.description is not None
                   and genie_response.result is not None
                   and genie_response != []
                else None
            )
            return GenieToolResponse(
                text=text_response,
                conversation_id=genie_response.conversation_id,
                query=genie_response.query,
                data=parsed_data,
            )
        except json.JSONDecodeError as e:
            return GenieToolResponse(
                text=f"{text_response} Data was returned but was unable to be parsed.",
                conversation_id=genie_response.conversation_id,
                query=genie_response.query,
                data=None,
            )

@mlflow.trace(span_type="TOOL")
class GenieTool(dspy.Module):

    def __init__(
            self,
            genie_space_id: str,
            genie_description: Optional[str] = None,
            client: Optional[WorkspaceClient] = None
    ):
        super().__init__()
        self.genie = Genie(space_id=genie_space_id, client=client)
        self.__doc__ = genie_description or getattr(self.genie, "description", None)

    @mlflow.trace()
    def forward(self, question: str, conversation_id: Optional[str] = None):
        genie_response = self.genie.ask_question(question, conversation_id, result_as_json=True)
        return GenieToolResponse.from_genie_response(genie_response)


class GenieAgentSignature(dspy.Signature):
    question: str = dspy.InputField()
    genie_text: str = dspy.InputField(desc="The text response from the Genie")
    genie_query: Optional[str] = dspy.InputField(
        desc=(
            "The SQL query used by the genie. Useful for explaining to the user exactly "
            "how the genie arrived at a specific answer"
        )
    )
    genie_data: Optional[list[dict[str, Any]]] = dspy.InputField(
        desc="The data returned by the genie."
    )
    answer: str = dspy.OutputField(
        desc=(
            "The answer to the user's question, or follow up questions "
            "if the genie needs additional information to continue"
        )
    )

@mlflow.trace(span_type="AGENT")
class GenieAgent(dspy.Module):

    # TODO: allow a dspy.LM instance to be given
    # TODO: allow a dspy.Module instance to be given
    # TODO: reference dspy.retrievers.databricks_rm
    def __init__(
            self,
            genie_space_id: str,
            lm_serving_endpoint_name: str,
            genie_description: Optional[str] = None,
            client: Optional[WorkspaceClient] = None
    ):
        super().__init__()
        self.genie = GenieTool(genie_space_id, genie_description, client)
        self.conversation_id = None
        self.__doc__ = genie_description or getattr(self.genie, "__doc__", None)
        self.lm = DatabricksLM(f"databricks/{lm_serving_endpoint_name}")
        self.generator = dspy.ChainOfThought(GenieAgentSignature)

    def forward(self, user_input: str):
        # ask the genie first
        genie_response = self.genie(user_input, self.conversation_id)

        if genie_response.conversation_id:
            self.conversation_id = genie_response.conversation_id

        # send the genie response through the lm
        with dspy.context(lm=self.lm):
            response = self.generator(
                **{
                    "question": user_input,
                    "genie_text": genie_response.text,
                    "genie_query": genie_response.query,
                    "genie_data": genie_response.data,
                }
            )

        return dspy.Prediction(response=response)

    def start_over(self):
        self.conversation_id = None