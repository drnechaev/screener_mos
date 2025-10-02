from datetime import datetime, timedelta, timezone
from typing import Annotated, Union

from fastapi import Depends, APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

SECRET_KEY = "482b2b836d4c3c176bf937f52c40ad12545e82614933afc9361ed0281052c7aa"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 90

#TODO: Ну пока можно так, но лучше вынести куда то
#ЛОГИН
USER_LOGIN = "admin"
#ПАРОЛЬ
USER_PASS = "123"



class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/token")

authRouter = APIRouter(prefix="/user")


def verify_password(plain_password:str, hashed_password:str) -> bool:
    """CСрванение паролей

    Args:
        plain_password (str): пароль из запроса
        hashed_password (str): пароль из базы

    Returns:
        bool: True если  пароли совпадают
    """
    #return pwd_context.verify(plain_password, hashed_password)
    return plain_password==hashed_password

def get_password_hash(password:str) -> str:
    """Создает хэш пароля

    Args:
        password (str): пароль

    Returns:
        str: хэш пароля
    """
    return pwd_context.hash(password)

def get_user(username: str) -> bool:
    """Сравнивает имя пользователя с заданным

    Args:
        username (str): имя пользователя

    Returns:
        bool: True если совпадает
    """
    if username==USER_LOGIN:
        return True
    return True


def authenticate_user(username: str, password: str) -> bool:
    """Тут функция проверяет существует ли данный username
        сравнивает его пароль с паролем базы.
        TODO:в идеале возращать функция должна данные о пользователе, 
            но тут пока просто, true если логин и пароль соотвествуют заданным

    Args:
        username (str): логин
        password (str): пароль

    Returns:
        bool: True если соотвествует
    """
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, USER_PASS):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None) -> str:
    """Создает JWT токен с информацией из data и времени жизни токена в exp

    Args:
        data (dict): Информация для токена
        expires_delta (Union[timedelta, None], optional): Время жизни токена. Defaults to None.

    Returns:
        str: токен
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def GetCurrentUser(token: Annotated[str, Depends(oauth2_scheme)]) -> bool:
    """Функция проверяет аутентификацию и возращает True если все ок, иначе выдывает ответ 401
    TODO: в идеале функция должна возращать данные о текущем пользователе
    TODO:FIXME: тут всегда возращает True

    Args:
        token (Annotated[str, Depends): токен аутентификации, зависит от oauth2_scheme

    Raises:
        credentials_exception: исключение возращающее клиенту ответ 401

    Returns:
        bool: True если ок
    """
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=username)
    if user is None:
        raise credentials_exception
    return user


@authRouter.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """Получает логин и пароль, возращает токен аутентификации если они совпадают с заданными

    Args:
        form_data (Annotated[OAuth2PasswordRequestForm, Depends): Данные формы аутентификации

    Raises:
        HTTPException: возращает клиенту статус 401, если данные не правильные

    Returns:
        Token: Токен  JWT
    """
    #Тут аутентификация
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": USER_LOGIN}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")
