from contextlib import contextmanager
from sqlalchemy import create_engine, select, text, inspect
from sqlalchemy.orm import sessionmaker
from .models import Base, ResourceCounter


class Database:
    def __init__(self, settings, engine=None):
        # SQLite is possible only through explicitly injected test engines.
        if engine is None:
            settings.validate()
            url = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
            engine = create_engine(url, pool_pre_ping=True)
        self.engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    @contextmanager
    def session(self):
        with self._sessions() as session:
            with session.begin():
                yield session

    def create_schema(self):
        # Several independently supervised processes can start simultaneously.
        with self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                connection.execute(text("SELECT pg_advisory_xact_lock(724162603)"))
            Base.metadata.create_all(connection)
            # Additive v1 quota migration; existing data is accounted lazily
            # under the same project/global locks before it is exposed again.
            additions = {
                "projects": {"metadata_bytes": "BIGINT NOT NULL DEFAULT 0", "quota_version": "INTEGER NOT NULL DEFAULT 0",
                             "pending_mutation_id": "VARCHAR(36)"},
                "image_objects": {"thumbnail_bytes": "BIGINT NOT NULL DEFAULT 0"},
            }
            for table, columns in additions.items():
                existing = {item["name"] for item in inspect(connection).get_columns(table)}
                for name, definition in columns.items():
                    if name not in existing:
                        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}"))
            if connection.execute(select(ResourceCounter.id).where(ResourceCounter.id == "global")).scalar_one_or_none() is None:
                connection.execute(ResourceCounter.__table__.insert().values(id="global", storage_bytes=0, reserved_bytes=0))

    def global_lock(self, session):
        return session.execute(select(ResourceCounter).where(ResourceCounter.id == "global").with_for_update()).scalar_one()
