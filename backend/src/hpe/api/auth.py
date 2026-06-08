"""JWT authentication and user management for HPE API.

Provides registration, login, and token-based authentication
for the multi-tenant SaaS platform.

Usage:
    from hpe.api.auth import get_current_user
    @router.get("/me")
    def me(user: User = Depends(get_current_user)):
        return user
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# JWT handling
try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None  # type: ignore[assignment]

SECRET_KEY = os.getenv("HPE_JWT_SECRET", "hpe-dev-secret-change-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = int(os.getenv("HPE_JWT_EXPIRE_MINUTES", "1440"))  # 24h default

security = HTTPBearer(auto_error=False)


# --- Models ---

class User(BaseModel):
    id: str
    email: str
    name: str
    company: Optional[str] = None
    role: str = "user"  # user, admin
    is_active: bool = True


class UserRegister(BaseModel):
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=6)
    name: str = Field(..., min_length=1)
    company: Optional[str] = None


class UserLogin(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User


# --- In-memory user store (replace with DB in production) ---
# This is a simple implementation for development. In production,
# users would be stored in the PostgreSQL database.

_users: dict[str, dict] = {}
_passwords: dict[str, str] = {}


def _hash_password(password: str) -> str:
    """Simple password hashing. Use bcrypt in production."""
    import hashlib
    return hashlib.sha256((password + SECRET_KEY).encode()).hexdigest()


# --- Token functions ---

def create_token(user_id: str, email: str, role: str = "user") -> str:
    """Create a JWT access token."""
    if pyjwt is None:
        # Fallback: simple base64 token for dev without PyJWT
        import base64, json
        payload = {"sub": user_id, "email": email, "role": role}
        return base64.b64encode(json.dumps(payload).encode()).decode()

    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES),
        "iat": datetime.utcnow(),
    }
    return pyjwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    if pyjwt is None:
        import base64, json
        try:
            return json.loads(base64.b64decode(token))
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")

    try:
        return pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# --- Dependencies ---

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """FastAPI dependency to get the authenticated user.

    If no token is provided, returns a default anonymous user
    for backward compatibility during development.
    """
    if credentials is None:
        # Anonymous access for development
        return User(id="anonymous", email="dev@higra.com.br", name="Dev User", role="admin")

    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")

    if user_id and user_id in _users:
        data = _users[user_id]
        return User(**data)

    # Token valid but user not in store (e.g., after restart)
    return User(
        id=payload.get("sub", "unknown"),
        email=payload.get("email", ""),
        name=payload.get("email", "").split("@")[0],
        role=payload.get("role", "user"),
    )


def get_current_active_user(user: User = Depends(get_current_user)) -> User:
    """Ensure the user is active."""
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")
    return user


# --- Auth endpoints ---

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
def register(req: UserRegister) -> TokenResponse:
    """Register a new user account."""
    # Check duplicate email
    for u in _users.values():
        if u["email"] == req.email:
            raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    _users[user_id] = {
        "id": user_id,
        "email": req.email,
        "name": req.name,
        "company": req.company,
        "role": "user",
        "is_active": True,
    }
    _passwords[user_id] = _hash_password(req.password)

    token = create_token(user_id, req.email)

    return TokenResponse(
        access_token=token,
        expires_in=TOKEN_EXPIRE_MINUTES * 60,
        user=User(**_users[user_id]),
    )


@router.post("/login", response_model=TokenResponse)
def login(req: UserLogin) -> TokenResponse:
    """Authenticate and get an access token."""
    # Find user by email
    user_data = None
    user_id = None
    for uid, u in _users.items():
        if u["email"] == req.email:
            user_data = u
            user_id = uid
            break

    if not user_data or not user_id:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if _passwords.get(user_id) != _hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user_id, req.email, user_data.get("role", "user"))

    return TokenResponse(
        access_token=token,
        expires_in=TOKEN_EXPIRE_MINUTES * 60,
        user=User(**user_data),
    )


@router.get("/me", response_model=User)
def get_me(user: User = Depends(get_current_user)) -> User:
    """Get the current authenticated user."""
    return user
