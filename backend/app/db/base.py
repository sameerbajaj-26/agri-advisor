from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base; import models in app/db/models.py (later) so Alembic sees them."""
