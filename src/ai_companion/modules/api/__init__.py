import httpx
import json
from enum import Enum, auto
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class SearchType(str, Enum):
    """Tipos de entidades que se pueden buscar"""
    ZONES = "zones"
    PROJECTS = "projects"
    DEVELOPERS = "developers"
    POIS = "pois"

class APIClient:
    """Cliente para la API de búsqueda inmobiliaria"""
    
    def __init__(self):
        from ai_companion.settings import settings
        
        self.base_url = settings.API_URL
        self.api_key = settings.API_KEY
        self.search_endpoint = "/ia/search"
    
    async def search(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza una búsqueda en la API inmobiliaria con los parámetros dados.
        
        Args:
            search_params: Diccionario con los parámetros de búsqueda según el formato de SearchDto
                - nameQuery: Texto para buscar entidades por nombre
                - semanticQuery: Texto para búsqueda semántica
                - searchIn: Lista de tipos de entidades a buscar (zones, projects, etc.)
                - params: Objeto con filtros (bedrooms, minPrice, maxPrice, propertyType)
                - flexibleSearch: Si se debe permitir búsqueda flexible
                - includeExamples: Si se deben incluir ejemplos en la respuesta
        
        Returns:
            Diccionario con los resultados de la búsqueda
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Validar y normalizar parámetros
        normalized_params = self._normalize_search_params(search_params)
        
        try:
            url = f"{self.base_url}{self.search_endpoint}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, 
                    headers=headers,
                    json=normalized_params,
                    timeout=15.0  # Tiempo de espera más largo para búsquedas complejas
                )
                
                response.raise_for_status()
                return response.json()
                
        except httpx.RequestError as e:
            logger.error(f"Error de conexión a la API : {str(e)}")
            return {
                "success": False,
                "message": f"Error de conexión: {str(e)}"
            }
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Error en la API: {e.response.status_code} - {e.response.text}")
            return {
                "success": False,
                "message": f"Error {e.response.status_code}: {e.response.text}"
            }
        
        except Exception as e:
            logger.error(f"Error inesperado en la consulta a la API: {str(e)}")
            return {
                "success": False,
                "message": f"Error inesperado: {str(e)}"
            }
    
    def _normalize_search_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza los parámetros de búsqueda para asegurar que cumplen con el formato esperado.
        """
        normalized = {}
        
        # Parámetros de nivel superior
        if "nameQuery" in params:
            normalized["nameQuery"] = params["nameQuery"]
        
        if "semanticQuery" in params:
            normalized["semanticQuery"] = params["semanticQuery"]
        
        # Asegurar que searchIn es una lista de strings válidos
        if "searchIn" in params and isinstance(params["searchIn"], list):
            # Convertir los valores a las constantes del enum
            valid_types = [st for st in params["searchIn"] if st in [t.value for t in SearchType]]
            if valid_types:
                normalized["searchIn"] = valid_types
            else:
                # Si no hay tipos válidos, usar al menos uno por defecto
                normalized["searchIn"] = [SearchType.ZONES.value]
        else:
            # Valor por defecto si no se especifica
            normalized["searchIn"] = [SearchType.ZONES.value]
        
        # Parámetros específicos de búsqueda
        if "params" in params and isinstance(params["params"], dict):
            search_params = {}
            
            if "bedrooms" in params["params"] and params["params"]["bedrooms"] is not None:
                try:
                    search_params["bedrooms"] = int(params["params"]["bedrooms"])
                except (ValueError, TypeError):
                    pass  # Ignorar si no es convertible a int
            
            if "minPrice" in params["params"] and params["params"]["minPrice"] is not None:
                try:
                    search_params["minPrice"] = int(params["params"]["minPrice"])
                except (ValueError, TypeError):
                    pass
            
            if "maxPrice" in params["params"] and params["params"]["maxPrice"] is not None:
                try:
                    search_params["maxPrice"] = int(params["params"]["maxPrice"])
                except (ValueError, TypeError):
                    pass
            
            if "propertyType" in params["params"] and params["params"]["propertyType"]:
                search_params["propertyType"] = str(params["params"]["propertyType"])
            
            if search_params:
                normalized["params"] = search_params
        
        # Banderas de control
        if "flexibleSearch" in params:
            normalized["flexibleSearch"] = bool(params["flexibleSearch"])
        else:
            normalized["flexibleSearch"] = True  # Por defecto activada para mejorar resultados
        
        if "includeExamples" in params:
            normalized["includeExamples"] = bool(params["includeExamples"])
        else:
            normalized["includeExamples"] = True  # Por defecto activada para dar ejemplos
        
        return normalized