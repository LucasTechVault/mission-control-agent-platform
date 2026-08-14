from mission_control.inference.gateway import ModelGateway
from mission_control.inference.requests import (
    ModelMessage,
    ModelRequest
)
from mission_control.investigation.models import InvestigationBrief

class InvestigationBriefGenerationError(RuntimeError):
    """Mission Control could not produce a valid InvestigationBrief"""
    
class InvestigationBriefGenerator:
    """Generate a typed investigation brief using a ModelGateway."""

    def __init__(
        self,
        gateway: ModelGateway,
        ) -> None:
        self._gateway = gateway
    
    async def generate(
        self,
        *,
        objective: str,
        context: str,
    ) -> InvestigationBrief:
        
        req = ModelRequest(
            messages=[
                ModelMessage(
                    role="system",
                    content=(
                        "You are Mission Control's initial investigation analyst."
                        "Produce an InvestigationBrief using only information supplied by user."
                        "Do not invent evidence."
                        "Hypotheses must be presented as hypotheses, not as established facts."
                        "Recommended actions should focus on investigation and evidence gathering."
                        ""
                    )
                ),
                ModelMessage(
                    role="user",
                    content=(
                        f"Investigation Objective: \n"
                        f"{objective}\n\n"
                        f"Known context:\n"
                        f"{context}\n\n"
                        "Populate every field in the required InvestigationBrief schema."
                    )
                )
            ],
            temperature=0.2,
            top_p=0.8,
            max_tokens=768,
            enable_thinking=False,
            response_schema=(
                InvestigationBrief.model_json_schema()
            ),
            response_schema_name="investigation-brief",
        )
        
        res = await self._gateway.generate(req)
        
        if res.text is None:
            raise InvestigationBriefGenerationError(
                "Model returned no final content."
            )
        
        try:
            return InvestigationBrief.model_validate_json(
                res.text
            )
        
        except ValueError as exc:
            raise InvestigationBriefGenerationError(
                "Model response could not be validated as an InvestigationBrief."
            ) from exc