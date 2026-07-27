"""
Shared exception types used across the agent and service layers.

Keeping exceptions in models/ avoids circular imports: services can raise
ArticleValidationError without importing from agents/.
"""


class ArticleValidationError(ValueError):
    """
    Raised when an ArticleRequest or planner output fails pre-generation validation.

    Callers should display the message directly — it is written for end-user consumption.
    """
    pass
