"""Provision a brand-new Postgres database + owning role for one collaborator,
inside your existing Postgres server (e.g. one Neon project can hold many
databases). Each user gets their own database — not a shared schema with
row-level filtering — so there's no cross-user query surface at all: their
role has zero grants on any other database, and Postgres enforces that at
the connection/privilege level, not the application.

Because the new role *owns* its database, TokenSense's normal schema
bootstrap (CREATE EXTENSION, CREATE TABLE) just works for them like it does
for you — no special config needed on their end.

Requires the executing connection's role to have CREATEROLE and CREATEDB
(Neon's default project owner role has both).

Idempotent-ish: re-running with the same db-name regenerates that role's
password (use --password to pin one) but will not touch an already-created
database's contents.

Usage:
    python scripts/create_user_database.py <owner-db-url> <db-name> [--role-name NAME] [--password PASSWORD]

Prints the resulting connection string. Hand that — and only that — to the
collaborator; it grants them nothing beyond their own database.
"""
from __future__ import annotations

import argparse
import re
import secrets

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


def _sanitize_identifier(name: str, prefix: str) -> str:
    slug = re.sub(r"[^a-z0-9_]", "_", name.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return f"{prefix}_{slug}" if slug else f"{prefix}_default"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("db_url", help="Owner connection string (must have CREATEDB + CREATEROLE)")
    parser.add_argument("db_name", help="Short label for the collaborator (e.g. their name or project) — used to derive the database and role names")
    parser.add_argument("--role-name", default=None, help="Postgres role name (default: derived from db-name)")
    parser.add_argument("--password", default=None, help="Role password (default: securely generated)")
    args = parser.parse_args()

    role_name = args.role_name or _sanitize_identifier(args.db_name, "tsuser")
    database_name = _sanitize_identifier(args.db_name, "tsdb")
    password = args.password or secrets.token_urlsafe(24)

    owner_url = make_url(args.db_url)
    engine = create_engine(owner_url)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        role_exists = conn.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": role_name}
        ).scalar()
        if role_exists:
            conn.execute(text(f'ALTER ROLE "{role_name}" WITH LOGIN PASSWORD :pw'), {"pw": password})
        else:
            conn.execute(
                text(
                    f'CREATE ROLE "{role_name}" WITH LOGIN PASSWORD :pw '
                    "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION"
                ),
                {"pw": password},
            )

        db_exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": database_name}
        ).scalar()
        if db_exists:
            print(f"Database {database_name!r} already exists — leaving its contents untouched.")
        else:
            # CREATE DATABASE ... OWNER requires the executing role to be
            # able to SET ROLE to the target, so grant membership first.
            current_user = conn.execute(text("SELECT current_user")).scalar()
            conn.execute(text(f'GRANT "{role_name}" TO "{current_user}"'))
            conn.execute(text(f'CREATE DATABASE "{database_name}" OWNER "{role_name}"'))

    friend_url = owner_url.set(username=role_name, password=password, database=database_name)
    print(f"Role ready:     {role_name}")
    print(f"Database ready: {database_name}")
    print()
    print("Give your collaborator this connection string (and only this):")
    print(f"  {friend_url.render_as_string(hide_password=False)}")
    print()
    print(
        "They put that in their own .mcp.json as TOKENSENSE_DB_URL — no other config needed, "
        "their role owns this database so schema bootstrap works normally for them."
    )


if __name__ == "__main__":
    main()
