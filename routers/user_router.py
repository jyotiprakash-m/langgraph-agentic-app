from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import SQLModel, Field, Session, select, func, desc
from pydantic import BaseModel
from utils.database import get_session

# ===========================
# User Table & Models
# ===========================

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    full_name: str
    username: str = Field(index=True, unique=True)
    password: str  # In production, store hashed passwords!
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc), 
        nullable=False
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc), 
        sa_column_kwargs={"onupdate": func.now()}, 
        nullable=False
    )

class UserCreate(BaseModel):
    full_name: str
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    full_name: str
    username: str
    created_at: datetime
    updated_at: datetime

class UserUpdatePassword(BaseModel):
    password: str

# ===========================
# API Router
# ===========================

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/", response_model=UserOut)
def create_user(
    user: UserCreate,
    session: Session = Depends(get_session)
):
    return create_user_service(session, user)

@router.get("/", response_model=List[UserOut])
def get_all_users(
    session: Session = Depends(get_session)
):
    return fetch_all_users(session)

@router.get("/{username}", response_model=UserOut)
def get_user_by_username(
    username: str,
    session: Session = Depends(get_session)
):
    return fetch_user_by_username(session, username)

@router.patch("/{username}/password", response_model=UserOut)
def update_user_password(
    username: str,
    password_update: UserUpdatePassword,
    session: Session = Depends(get_session)
):
    return update_user_password_service(session, username, password_update.password)

@router.delete("/{username}")
def delete_user(
    username: str,
    session: Session = Depends(get_session)
):
    return delete_user_service(session, username)

# ===========================
# Service Functions
# ===========================

def create_user_service(session: Session, user_data: UserCreate) -> User:
    """
    Create a new user. Username must be unique.
    """
    existing = session.exec(
        select(User).where(User.username == user_data.username)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"User with username '{user_data.username}' already exists.")
    new_user = User(
        full_name=user_data.full_name,
        username=user_data.username,
        password=user_data.password,  # In production, hash the password!
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user

def fetch_all_users(session: Session) -> List[User]:
    """
    Fetch all users, ordered by created_at descending.
    """
    query = select(User).order_by(desc(User.created_at))
    return list(session.exec(query).all())

def fetch_user_by_username(session: Session, username: str) -> Optional[User]:
    """
    Fetch a user by username.
    """
    user = session.exec(
        select(User).where(User.username == username)
    ).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with username '{username}' not found.")
    return user

def update_user_password_service(session: Session, username: str, new_password: str) -> User:
    """
    Update a user's password.
    """
    user = session.exec(
        select(User).where(User.username == username)
    ).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with username '{username}' not found.")
    user.password = new_password  # In production, hash the password!
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

def delete_user_service(session: Session, username: str) -> dict:
    """
    Delete a user by username.
    """
    user = session.exec(
        select(User).where(User.username == username)
    ).first()
    if user:
        session.delete(user)
        session.commit()
        return {"detail": f"User '{username}' deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"User with username '{username}' not found.")