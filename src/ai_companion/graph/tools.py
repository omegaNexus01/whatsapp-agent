# En tools.py - Versión con contexto global
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import contextvars
from typing import Optional,List
import json

class SearchProjectsDto(BaseModel):
    name_query: Optional[str] = Field(default=None, description="Búsqueda por nombre específico del proyecto")
    semantic_query: Optional[str] = Field(default=None, description="Búsqueda semántica en lenguaje natural")
    search_in: List[str] = Field(default=["projects"], description="Dónde buscar: projects, zones, developers, pois")
    bedrooms: Optional[int] = Field(default=None, description="Número de habitaciones")
    min_price: Optional[int] = Field(default=None, description="Precio mínimo")
    max_price: Optional[int] = Field(default=None, description="Precio máximo")
    property_type: Optional[str] = Field(default=None, description="Tipo de propiedad: apartment, house, etc.")
    flexible_search: bool = Field(default=True, description="Permitir búsqueda flexible")
    use_name_guesser: bool = Field(default=True, description="Usar NameGuesser para mejores resultados")

class SendProjectCardDto(BaseModel):
    project_id: int = Field(description="ID numérico del proyecto encontrado en los resultados de búsqueda")
    project_name: str = Field(description="Nombre del proyecto para confirmar al usuario")
    
class SendUnitCardDto(BaseModel):
    unit_id: str = Field(description="ID de la unidad encontrada en los resultados de búsqueda (formato UUID)")
    unit_name: str = Field(description="Nombre de la unidad para confirmar al usuario")    

# Variable de contexto para thread_id
current_thread_id = contextvars.ContextVar('thread_id', default='unknown')

@tool(
    "send_project_card",
    args_schema=SendProjectCardDto
)
async def send_project_card(project_id: int, project_name: str) -> str:
    """
    Envía una tarjeta de proyecto inmobiliario al usuario vía WhatsApp.
    
    Usar cuando el usuario pida información específica sobre UN proyecto.
    """
    try:
        from ai_companion.modules.api import APIClient
        
        # Obtener thread_id del contexto
        thread_id = current_thread_id.get()
        
        api_client = APIClient()
        result = await api_client.send_project_card(project_id, thread_id)
        
        if result.get("success", False):
            return "Su tarjeta del proyecto ha sido enviada exitosamente."
        else:
            error_msg = result.get('message', 'Error desconocido')
            print(f"❌ Error enviando tarjeta: {error_msg}")
            return f"❌ Hubo un problema enviando la tarjeta del proyecto {project_name}."
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in send_project_card: {str(e)}")
        return f"❌ Error técnico: {str(e)}"

@tool(
    "send_unit_card",
    args_schema=SendUnitCardDto
)
async def send_unit_card(unit_id: str, unit_name: str) -> str:
    """
    Envía una tarjeta de unidad específica al usuario vía WhatsApp.
    
    Usar cuando el usuario pida información sobre UNA unidad en un proyecto.
    Las unidades tienen IDs en formato UUID (ej: "46831b47-c4e5-451d-8b74-6058ebbf639b").
    """
    print(f"🔧 EJECUTANDO: send_unit_card")
    print(f"   📝 Parámetros: unit_id={unit_id}, unit_name={unit_name}")
    
    try:
        from ai_companion.modules.api import APIClient
        
        # Obtener thread_id del contexto
        thread_id = current_thread_id.get()
        print(f"   🔗 thread_id: {thread_id}")
        
        api_client = APIClient()
        result = await api_client.send_unit_card(unit_id, thread_id)
        
        if result.get("success", False):
            print(f"✅ Tarjeta de unidad enviada exitosamente")
            return "Su tarjeta de unidad ha sido enviada exitosamente."
        else:
            error_msg = result.get('message', 'Error desconocido')
            print(f"❌ Error enviando tarjeta de unidad: {error_msg}")
            return f"❌ Hubo un problema enviando la tarjeta de la unidad {unit_name}."
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in send_unit_card: {str(e)}")
        return f"❌ Error técnico: {str(e)}"     
      
@tool(
    "search_projects",
    args_schema=SearchProjectsDto
)
async def search_projects(
    name_query: Optional[str] = None,
    semantic_query: Optional[str] = None, 
    search_in: List[str] = ["projects"],
    bedrooms: Optional[int] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    property_type: Optional[str] = None,
    flexible_search: bool = True,
    use_name_guesser: bool = True
) -> str:
    """
    Busca proyectos en la base de datos.
    
    Usar cuando el usuario:
    - Pregunte sobre proyectos disponibles
    - Busque por ubicación, nombre, o características específicas
    - Quiera comparar opciones
    - Haga consultas sobre precios o disponibilidad
    - Quiera unidades específicas dentro de un proyecto
    
    
    Args:
        name_query: Nombre específico del proyecto o desarrollador
        semantic_query: Descripción en lenguaje natural de lo que busca
        search_in: Lista de categorías donde buscar (projects, zones, developers, pois)
        bedrooms: Número de habitaciones
        min_price: Precio mínimo
        max_price: Precio máximo  
        property_type: Tipo de propiedad
        flexible_search: Si permitir búsqueda flexible
        use_name_guesser: Si usar NameGuesser para mejores resultados
    """
    try:
        from ai_companion.modules.api import APIClient
        
        # Construir parámetros de búsqueda exactamente como en search_node
        search_params = {
            "nameQuery": name_query,
            "semanticQuery": semantic_query,
            "searchIn": search_in or ["projects"],
            "params": {},
            "flexibleSearch": flexible_search,
            "useNameGuesser": use_name_guesser
        }
        
        # Agregar parámetros opcionales en el objeto params
        if bedrooms is not None:
            search_params["params"]["bedrooms"] = bedrooms
        if min_price is not None:
            search_params["params"]["minPrice"] = min_price
        if max_price is not None:
            search_params["params"]["maxPrice"] = max_price
        if property_type is not None:
            search_params["params"]["propertyType"] = property_type
        
        print(f"🔍 Parámetros de búsqueda: {search_params}")
        
        api_client = APIClient()
        results = await api_client.search(search_params)
        
        print(f"🔍 Resultados de búsqueda: {results}")
        
        if results.get("success", False):
            # Usar la misma función de formateo que search_node
            formatted_results = await format_search_results(results, search_params)
            return formatted_results
        else:
            return f"❌ Error en la búsqueda: {results.get('message', 'Error desconocido')}"
            
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error en search_real_estate_projects: {str(e)}")
        return f"❌ Error técnico en la búsqueda: {str(e)}"

async def format_search_results(search_results, search_params):
    """Formats the search results with CLEAR unit IDs visible"""
    try:
        if not search_results.get("success", False):
            return f"Error in search: {search_results.get('message', 'Unknown error')}"
        
        projects_data = search_results.get("projects", {})
        
        if not projects_data or not projects_data.get("regions"):
            return "No projects found matching your criteria."
        
        # Start building formatted response
        formatted = "🏠 **SEARCH RESULTS** 🏠\n\n"
        
        # Add search context
        if search_params.get("nameQuery"):
            formatted += f"🔍 **Searched for:** {search_params['nameQuery']}\n"
        if search_params.get("semanticQuery"):
            formatted += f"🎯 **Query:** {search_params['semanticQuery']}\n"
        
        formatted += "\n---\n\n"
        
        # Process each region
        for region in projects_data["regions"]:
            region_name = region.get("regionName", "Unknown Region")
            counts = region.get("counts", {})
            
            formatted += f"📍 **{region_name}**\n"
            formatted += f"   • {counts.get('totalAssociatedProjectsCount', 0)} total projects available\n"
            formatted += f"   • {counts.get('projectsMatchingExactParamsCount', 0)} match your criteria\n\n"
            
            # Show exact matches with CLEAR UNIT IDs
            exact_matches = region.get("exactMatchExamples", [])
            if exact_matches:
                formatted += "🎯 **Perfect Matches:**\n"
                for i, project in enumerate(exact_matches[:3], 1):
                    project_name = project.get("projectName", "Unnamed Project")
                    project_id = project.get("projectId", "")
                    units_count = project.get("unitsMatchingParamsCount", 0)
                    
                    formatted += f"   {i}. **{project_name}** (Project ID: {project_id})\n"
                    formatted += f"      • {units_count} units match your criteria\n"
                    
                    # ✅ MOSTRAR UNIT IDs MUY CLARAMENTE
                    units_example = project.get("unitsExample", [])
                    if units_example:
                        formatted += f"      📋 **AVAILABLE UNITS:**\n"
                        for unit in units_example[:3]:  # Mostrar hasta 3 unidades
                            unit_id = unit.get("unitId", "")
                            unit_name = unit.get("unitName", "")
                            bedrooms = unit.get("bedrooms", "")
                            price = unit.get("price", "")
                            unit_type = unit.get("unitType", "")
                            
                            formatted += f"         • **{unit_name}**\n"
                            formatted += f"           🆔 Unit ID: {unit_id}\n"  # ✅ MUY CLARO
                            formatted += f"           🛏️ {bedrooms} bed {unit_type}\n"
                            if price:
                                formatted += f"           💰 ${price:,}\n"
                            formatted += "\n"
                    formatted += "\n"
            
            formatted += "---\n\n"
        
        # ✅ INSTRUCCIONES MUY CLARAS PARA EL AGENTE
        formatted += "💡 **IMPORTANT FOR AGENT:**\n"
        formatted += "• To send project card: Use the Project ID (numeric)\n"
        formatted += "• To send unit card: Use the EXACT Unit ID shown above\n"
        formatted += "• DO NOT invent or modify Unit IDs\n\n"
        
        # Include raw data for agent context
        formatted += f"📊 **Raw Data:**\n```json\n{json.dumps(search_results, indent=2)}\n```"
        
        return formatted
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error formatting search results: {str(e)}")
        return f"Error formatting results: {str(e)}\n\nRaw data: {search_results}"