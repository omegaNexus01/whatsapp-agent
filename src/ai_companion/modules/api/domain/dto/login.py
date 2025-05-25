"""
Módulo define los DTOs para la autenticación de usuarios.
"""

from pydantic import BaseModel, Field
from uuid import UUID


class LoginDto(BaseModel):
    """
    DTO para la petición de autenticación de usuarios.
    """

    preferredUsername: str
    password: str


class AuthenticationResult(BaseModel):
    """
    DTO para el resultado de la autenticación.
    """

    access_token: str = Field(..., alias="AccessToken")
    expires_in: int = Field(..., alias="ExpiresIn")
    id_token: str = Field(..., alias="IdToken")
    refresh_token: str = Field(..., alias="RefreshToken")
    token_type: str = Field(..., alias="TokenType")


class ChallengeParameters(BaseModel):
    """
    DTO para los parámetros del desafío.
    """

    pass


class Metadata(BaseModel):
    """
    DTO para los metadatos de la respuesta.
    """

    http_status_code: int = Field(..., alias="httpStatusCode")
    request_id: UUID = Field(..., alias="requestId")
    attempts: int
    total_retry_delay: int = Field(..., alias="totalRetryDelay")


class LoginResponseDto(BaseModel):
    """
    DTO para la respuesta del login de los usuarios.
    """

    metadata: Metadata = Field(..., alias="$metadata")
    authentication_result: AuthenticationResult = Field(
        ..., alias="AuthenticationResult"
    )
    challenge_parameters: ChallengeParameters = Field(..., alias="ChallengeParameters")
