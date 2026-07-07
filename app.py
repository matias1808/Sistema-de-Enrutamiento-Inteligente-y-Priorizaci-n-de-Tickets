import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from datetime import datetime
import io
import re


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Gestión automatizada de tickets con Machine Learning",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_TITLE = "Gestión automatizada de tickets con Machine Learning"


# ============================================================
# ARCHIVOS BASE
# ============================================================

BASE_TICKETS_PATHS = [
    "base_tickets (1).xlsx",
    "base_tickets.xlsx",
    "base_tickets.csv"
]

BD_USERS_PATHS = [
    "BD_USERS (1).xlsx",
    "BD_USERS.xlsx",
    "BD_USERS.csv"
]


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 34px;
            font-weight: 800;
            color: #12355B;
            margin-bottom: 0px;
        }

        .subtitle {
            font-size: 16px;
            color: #4f5b66;
            margin-bottom: 24px;
        }

        .metric-card {
            padding: 18px;
            border-radius: 16px;
            background: #f7f9fc;
            border: 1px solid #e5e9f0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }

        .small-note {
            font-size: 13px;
            color: #6c757d;
        }

        .ok-box {
            padding: 14px;
            background: #ecfdf3;
            border: 1px solid #abefc6;
            border-radius: 14px;
            color: #067647;
            font-size: 14px;
        }

        .warn-box {
            padding: 14px;
            background: #fffaeb;
            border: 1px solid #fedf89;
            border-radius: 14px;
            color: #93370d;
            font-size: 14px;
        }

        .powerbi-box {
            padding: 12px;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def header():
    st.markdown(f"<div class='main-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='subtitle'>
        Clasificación automática de tickets mediante conversión semántica y reglas de Machine Learning simuladas.
        </div>
        """,
        unsafe_allow_html=True
    )


def kpi_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:13px;color:#6c757d;">{label}</div>
            <div style="font-size:30px;font-weight:800;color:#12355B;">{value}</div>
            <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def buscar_archivo(posibles_nombres):
    for nombre in posibles_nombres:
        if Path(nombre).exists():
            return nombre
    return None


def leer_dataframe(uploaded_file, posibles_nombres):
    if uploaded_file is not None:
        nombre = uploaded_file.name

        if nombre.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        return df, nombre

    ruta = buscar_archivo(posibles_nombres)

    if ruta is None:
        return pd.DataFrame(), None

    if ruta.lower().endswith(".csv"):
        df = pd.read_csv(ruta)
    else:
        df = pd.read_excel(ruta)

    return df, ruta


def limpiar_columnas(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalizar_columna(df, posibles_nombres, nombre_final):
    df = df.copy()

    for col in posibles_nombres:
        if col in df.columns:
            df = df.rename(columns={col: nombre_final})
            return df

    return df


def obtener_signature_archivos(tickets_file, users_file):
    ticket_sig = None
    users_sig = None

    if tickets_file is not None:
        ticket_sig = (tickets_file.name, tickets_file.size)

    if users_file is not None:
        users_sig = (users_file.name, users_file.size)

    return str(ticket_sig), str(users_sig)


# ============================================================
# CARGA DE BASES
# ============================================================

def cargar_bd_users(uploaded_file=None):
    df, fuente = leer_dataframe(uploaded_file, BD_USERS_PATHS)

    if df.empty:
        return pd.DataFrame(), fuente

    df = limpiar_columnas(df)

    df = normalizar_columna(df, ["Matrícula", "Matricula", "MATRICULA"], "Matricula")
    df = normalizar_columna(df, ["Nombre", "NOMBRE"], "Nombre")
    df = normalizar_columna(df, ["Área", "Area", "AREA"], "Area")
    df = normalizar_columna(df, ["DNI", "Documento"], "DNI")
    df = normalizar_columna(df, ["Telefono", "Teléfono", "TELEFONO"], "Telefono")
    df = normalizar_columna(df, ["Correo", "Email", "EMAIL"], "Correo")

    columnas_necesarias = ["Matricula", "Nombre", "Area", "DNI", "Telefono", "Correo"]

    for col in columnas_necesarias:
        if col not in df.columns:
            df[col] = "No disponible"

    df["Matricula"] = df["Matricula"].astype(str).str.strip().str.upper()
    df["Nombre"] = df["Nombre"].fillna("No disponible")
    df["Area"] = df["Area"].fillna("No disponible")
    df["Correo"] = df["Correo"].fillna("No disponible")

    return df[columnas_necesarias], fuente


def cargar_base_tickets(uploaded_file=None):
    df, fuente = leer_dataframe(uploaded_file, BASE_TICKETS_PATHS)

    if df.empty:
        return pd.DataFrame(), fuente

    df = limpiar_columnas(df)

    df = normalizar_columna(df, ["Matrícula", "Matricula", "MATRICULA"], "Matricula")
    df = normalizar_columna(df, ["Descripción", "Descripcion", "DESCRIPCION"], "Descripcion")
    df = normalizar_columna(df, ["Título", "Titulo", "TITULO"], "Titulo")
    df = normalizar_columna(df, ["Prioridad", "PRIORIDAD"], "Prioridad")
    df = normalizar_columna(df, ["Creado", "Fecha Creado", "Fecha"], "Creado")
    df = normalizar_columna(df, ["Asignado", "Grupo Asignado"], "Asignado")
    df = normalizar_columna(df, ["Estado", "ESTADO"], "Estado")

    columnas_necesarias = [
        "Matricula",
        "Ticket",
        "Prioridad",
        "Titulo",
        "Descripcion",
        "Creado",
        "Asignado",
        "Estado"
    ]

    for col in columnas_necesarias:
        if col not in df.columns:
            df[col] = np.nan

    df["Matricula"] = df["Matricula"].astype(str).str.strip().str.upper()
    df["Ticket"] = df["Ticket"].fillna("SIN_TICKET")
    df["Prioridad"] = df["Prioridad"].fillna("Sin clasificar")
    df["Titulo"] = df["Titulo"].fillna("Sin título")
    df["Descripcion"] = df["Descripcion"].fillna("Sin descripción")
    df["Creado"] = df["Creado"].fillna("No disponible")
    df["Asignado"] = df["Asignado"].fillna("Pendiente asignación")
    df["Estado"] = df["Estado"].fillna("Pendiente")

    return df[columnas_necesarias], fuente


def cruzar_tickets_con_usuarios(tickets, usuarios):
    if tickets.empty:
        return tickets

    tickets = tickets.copy()

    if usuarios.empty:
        tickets["Nombre"] = "No disponible"
        tickets["Area"] = "No disponible"
        tickets["Correo"] = "No disponible"
        return tickets

    tickets = tickets.merge(
        usuarios[["Matricula", "Nombre", "Area", "Correo"]],
        on="Matricula",
        how="left"
    )

    tickets["Nombre"] = tickets["Nombre"].fillna("No encontrado")
    tickets["Area"] = tickets["Area"].fillna("No encontrado")
    tickets["Correo"] = tickets["Correo"].fillna("No encontrado")

    return tickets


# ============================================================
# CONVERSIÓN SEMÁNTICA MEJORADA
# ============================================================

DICCIONARIO_SEMANTICO = {
    "Accesos y permisos": [
        "acceso", "accesos", "login", "logueo", "contraseña", "password",
        "clave", "permiso", "permisos", "credencial", "credenciales",
        "usuario", "usuarios", "ingresar", "habilitar", "deshabilitar",
        "desbloqueo", "bloqueo", "cuenta bloqueada", "restablecer",
        "autenticación", "autenticacion", "mfa", "token", "perfil",
        "rol", "roles", "privilegio", "privilegios"
    ],
    "Incidente o falla": [
        "error", "errores", "falla", "fallas", "incidente", "incidencia",
        "problema", "no funciona", "no carga", "no responde", "caída",
        "caida", "bloqueo", "diferencia", "validación", "validacion",
        "revisar", "inconsistencia", "inconsistencias", "corregir",
        "corrección", "correccion", "bug", "defecto", "intermitente",
        "mensaje de error", "pantalla en blanco", "se cuelga", "se cae",
        "no permite", "rechaza", "fallido", "fallida"
    ],
    "Solicitud de servicio": [
        "solicitud", "solicito", "requerimiento", "requerir", "alta",
        "baja", "asignación", "asignacion", "reasignación", "reasignacion",
        "instalación", "instalacion", "instalar", "licencia", "licencias",
        "software", "actualización", "actualizacion", "carga", "subir",
        "modificar", "crear", "eliminar", "agregar", "activar",
        "desactivar", "configurar", "habilitación", "habilitacion",
        "aprobar", "aprobación", "aprobacion"
    ],
    "Aplicaciones y sistemas": [
        "sistema", "sistemas", "aplicación", "aplicacion", "app",
        "plataforma", "herramienta", "archivo", "archivos", "reporte",
        "reportes", "tipo de cambio", "cálculo", "calculo", "datos",
        "base de datos", "bd", "campaña", "dashboard", "power bi",
        "excel", "macro", "servicenow", "teams", "sharepoint",
        "sap", "crm", "erp", "portal", "web", "módulo", "modulo"
    ],
    "Rendimiento": [
        "lento", "lenta", "lentitud", "demora", "demorado", "demorada",
        "rendimiento", "performance", "tarda", "cae", "intermitente",
        "congelado", "congelada", "se demora", "demasiado lento",
        "tiempo de respuesta", "timeout", "latencia", "saturado",
        "saturación", "saturacion"
    ],
    "Cambio o mantenimiento": [
        "cambio", "cambios", "mantenimiento", "configuración",
        "configuracion", "parámetro", "parametro", "actualizar",
        "ajuste", "programado", "despliegue", "release", "versión",
        "version", "parche", "upgrade", "migración", "migracion",
        "mejora", "implementación", "implementacion"
    ],
    "Urgencia o impacto": [
        "urgente", "urgencia", "crítico", "critico", "crítica", "critica",
        "impacto", "impacta", "masivo", "masiva", "bloqueante",
        "prioridad", "producción", "produccion", "afecta", "afectación",
        "afectacion", "varios usuarios", "muchos usuarios", "todos los usuarios",
        "no se puede operar", "paralizado", "paralizada", "operación detenida",
        "operacion detenida", "alto impacto", "cliente afectado", "clientes afectados"
    ],
    "Datos y reportes": [
        "reporte", "reportes", "data", "datos", "registro", "registros",
        "tabla", "tablas", "campo", "campos", "base", "base de datos",
        "excel", "archivo", "carga masiva", "validar datos", "cuadre",
        "descuadre", "monto", "importes", "indicador", "kpi", "métrica",
        "metrica", "dashboard", "power bi"
    ]
}


def normalizar_texto(texto):
    return str(texto).lower().strip()


def contar_dimension(texto, palabras):
    texto = normalizar_texto(texto)
    return sum(1 for palabra in palabras if palabra in texto)


def obtener_categoria_semantica(titulo, descripcion):
    texto = normalizar_texto(f"{titulo} {descripcion}")

    puntajes = {}

    for categoria, palabras in DICCIONARIO_SEMANTICO.items():
        puntajes[categoria] = contar_dimension(texto, palabras)

    categoria_final = max(puntajes, key=puntajes.get)

    if puntajes[categoria_final] == 0:
        return "General"

    return categoria_final


def extraer_variables_semanticas(titulo, descripcion):
    texto = normalizar_texto(f"{titulo} {descripcion}")

    variables = {
        "Cantidad_Palabras": len(texto.split()),
        "Longitud_Texto": len(texto)
    }

    total = 0

    for categoria, palabras in DICCIONARIO_SEMANTICO.items():
        valor = contar_dimension(texto, palabras)
        variables[f"Sem_{categoria}"] = valor
        total += valor

    variables["Total_Indicadores_Semanticos"] = total
    variables["Categoria_Semantica"] = obtener_categoria_semantica(titulo, descripcion)

    return variables


# ============================================================
# MODELO DE CLASIFICACIÓN AUTOMÁTICA MEJORADO
# ============================================================

def clasificar_ticket(titulo, descripcion, estado="Pendiente", es_critico=False):
    texto = normalizar_texto(f"{titulo} {descripcion}")
    variables = extraer_variables_semanticas(titulo, descripcion)

    score = 0

    score += variables.get("Sem_Urgencia o impacto", 0) * 3.2
    score += variables.get("Sem_Incidente o falla", 0) * 2.4
    score += variables.get("Sem_Rendimiento", 0) * 1.8
    score += variables.get("Sem_Accesos y permisos", 0) * 1.5
    score += variables.get("Sem_Aplicaciones y sistemas", 0) * 1.2
    score += variables.get("Sem_Datos y reportes", 0) * 1.2
    score += variables.get("Sem_Solicitud de servicio", 0) * 0.9
    score += variables.get("Sem_Cambio o mantenimiento", 0) * 0.8

    palabras_n3 = [
        "crítico", "critico", "crítica", "critica", "masivo", "masiva",
        "bloqueante", "producción", "produccion", "caída", "caida",
        "impacto", "afectación", "afectacion", "sistema crítico",
        "plataforma crítica", "todos los usuarios", "operación detenida",
        "operacion detenida", "paralizado", "paralizada", "alto impacto",
        "cliente afectado", "clientes afectados", "no se puede operar"
    ]

    palabras_n2 = [
        "error", "falla", "diferencia", "validación", "validacion",
        "revisar", "inconsistencia", "inconsistencias", "tipo de cambio",
        "cálculo", "calculo", "actualización", "actualizacion",
        "corregir", "reporte", "reportes", "datos", "bd", "base de datos",
        "dashboard", "power bi", "servicenow", "sap", "crm", "erp",
        "intermitente", "latencia", "timeout", "descuadre", "cuadre"
    ]

    palabras_n1 = [
        "solicitud", "alta", "baja", "acceso", "permiso", "usuario",
        "carga", "subir", "instalación", "instalacion", "licencia",
        "campaña", "crear", "agregar", "activar", "desactivar",
        "restablecer contraseña", "desbloqueo"
    ]

    if es_critico:
        score += 5.5

    if any(p in texto for p in palabras_n3):
        score += 4.0

    if any(p in texto for p in palabras_n2):
        score += 1.8

    if str(estado).lower() in ["cerrado", "resuelto", "planificado"]:
        score -= 0.8

    # Prioridad predicha
    if es_critico or score >= 8:
        prioridad = "Crítica"
    elif score >= 5:
        prioridad = "Alta"
    elif score >= 2.2:
        prioridad = "Media"
    else:
        prioridad = "Baja"

    # Grupo asignado automático
    if es_critico or prioridad == "Crítica" or any(p in texto for p in palabras_n3):
        grupo_asignado = "Mesa de Ayuda N3"
        nivel_soporte = "Experto"
    elif prioridad == "Alta" or any(p in texto for p in palabras_n2):
        grupo_asignado = "Mesa de Ayuda N2"
        nivel_soporte = "Intermedio"
    else:
        grupo_asignado = "Mesa de Ayuda N1"
        nivel_soporte = "Inicial"

    # Confianza más alta y revisión humana menor
    total_senales = variables["Total_Indicadores_Semanticos"]

    if es_critico:
        confianza = 0.96
    elif total_senales >= 5:
        confianza = 0.94
    elif total_senales == 4:
        confianza = 0.91
    elif total_senales == 3:
        confianza = 0.88
    elif total_senales == 2:
        confianza = 0.82
    elif total_senales == 1:
        confianza = 0.76
    else:
        confianza = 0.70

    if prioridad in ["Crítica", "Alta"]:
        confianza += 0.02

    if grupo_asignado == "Mesa de Ayuda N3":
        confianza += 0.01

    confianza = min(confianza, 0.98)

    # Revisión humana reducida
    if confianza < 0.72:
        requiere_revision = "Sí"
        motivo_revision = "Baja evidencia semántica"
    elif prioridad == "Crítica" and not es_critico and confianza < 0.90:
        requiere_revision = "Sí"
        motivo_revision = "Caso crítico requiere validación"
    else:
        requiere_revision = "No"
        motivo_revision = "Clasificación automática confiable"

    prioridades = ["Crítica", "Alta", "Media", "Baja"]
    probabilidades = {}

    probabilidades[prioridad] = confianza

    restante = 1 - confianza
    otros = [p for p in prioridades if p != prioridad]

    for p in otros:
        probabilidades[p] = restante / len(otros)

    return {
        "Prioridad_Predicha": prioridad,
        "Grupo_Asignado_ML": grupo_asignado,
        "Nivel_Soporte": nivel_soporte,
        "Confianza_Modelo": round(confianza, 3),
        "Requiere_Revision": requiere_revision,
        "Motivo_Revision": motivo_revision,
        "Categoria_Semantica": variables["Categoria_Semantica"],
        "Probabilidades": probabilidades,
        "Variables": variables,
        "Score": round(score, 2),
        "Marcado_Critico": "Sí" if es_critico else "No"
    }


def aplicar_modelo_a_tickets(df):
    df = df.copy()

    resultados = []

    for _, row in df.iterrows():
        titulo = row.get("Titulo", "")
        descripcion = row.get("Descripcion", "")
        estado = row.get("Estado", "Pendiente")

        resultado = clasificar_ticket(
            titulo=titulo,
            descripcion=descripcion,
            estado=estado,
            es_critico=False
        )

        resultados.append(resultado)

    df["Categoria_Semantica"] = [r["Categoria_Semantica"] for r in resultados]
    df["Prioridad_Predicha"] = [r["Prioridad_Predicha"] for r in resultados]
    df["Grupo_Asignado_ML"] = [r["Grupo_Asignado_ML"] for r in resultados]
    df["Nivel_Soporte"] = [r["Nivel_Soporte"] for r in resultados]
    df["Confianza_Modelo"] = [r["Confianza_Modelo"] for r in resultados]
    df["Requiere_Revision"] = [r["Requiere_Revision"] for r in resultados]
    df["Motivo_Revision"] = [r["Motivo_Revision"] for r in resultados]
    df["Score_Modelo"] = [r["Score"] for r in resultados]
    df["Marcado_Critico"] = [r["Marcado_Critico"] for r in resultados]
    df["Estado_Publicacion"] = "Clasificado automáticamente"

    return df


# ============================================================
# ASISTENTE GRATUITO
# ============================================================

def respuesta_asistente(opcion):
    opcion = normalizar_texto(opcion)

    if "registro un ticket" in opcion:
        return (
            "Ingresa a 'Nuevo ticket', selecciona una matrícula, completa título y descripción. "
            "También puedes marcar el checkbox si el caso es crítico. Al clasificarlo, aparecerá en la bandeja."
        )

    if "modelo ml" in opcion:
        return (
            "El modelo usa conversión semántica. Analiza palabras asociadas a accesos, fallas, solicitudes, "
            "sistemas, reportes, rendimiento, cambios e impacto. Con eso define prioridad y nivel N1, N2 o N3."
        )

    if "bases usa" in opcion:
        return (
            "La aplicación solo usa dos archivos: base_tickets y BD_USERS. "
            "Con ambos genera la clasificación automática, bandeja y métricas."
        )

    if "niveles" in opcion:
        return (
            "N1 atiende casos simples u operativos. N2 atiende casos técnicos o funcionales intermedios. "
            "N3 atiende casos críticos, complejos o que requieren expertos."
        )

    if "revisión humana" in opcion or "revision humana" in opcion:
        return (
            "La revisión humana se activa solo si la confianza es baja o si el sistema detecta un caso crítico "
            "que requiere validación adicional."
        )

    if "descargo" in opcion:
        return "En 'Bandeja de tickets' puedes filtrar y descargar el resultado final en Excel."

    return "Selecciona una opción del asistente para ver la explicación."


def asistente_sidebar():
    st.sidebar.markdown("### 🤖 Asistente gratuito")

    opcion = st.sidebar.selectbox(
        "Ayuda rápida",
        [
            "Selecciona una opción",
            "¿Cómo registro un ticket?",
            "¿Cómo funciona el modelo ML?",
            "¿Qué bases usa la aplicación?",
            "¿Cómo funcionan los niveles N1, N2 y N3?",
            "¿Qué significa revisión humana?",
            "¿Cómo descargo los resultados?"
        ]
    )

    if st.sidebar.button("Consultar asistente", use_container_width=True):
        st.sidebar.success(respuesta_asistente(opcion))

    st.sidebar.divider()


# ============================================================
# CARGA DESDE SIDEBAR
# ============================================================

def cargar_archivos_sidebar():
    st.sidebar.markdown("### Cargar archivos")

    tickets_file = st.sidebar.file_uploader(
        "Base de tickets",
        type=["xlsx", "csv"],
        key="upload_tickets"
    )

    users_file = st.sidebar.file_uploader(
        "BD de usuarios",
        type=["xlsx", "csv"],
        key="upload_users"
    )

    return tickets_file, users_file


def procesar_datos(tickets_file, users_file, forzar=False):
    ticket_sig, users_sig = obtener_signature_archivos(tickets_file, users_file)
    nueva_signature = f"{ticket_sig}|{users_sig}"

    if (
        not forzar
        and "data_signature" in st.session_state
        and st.session_state.data_signature == nueva_signature
        and "tickets_ml" in st.session_state
        and "users" in st.session_state
    ):
        return

    users, ruta_users = cargar_bd_users(users_file)
    tickets, ruta_tickets = cargar_base_tickets(tickets_file)

    tickets_cruzados = cruzar_tickets_con_usuarios(tickets, users)

    if not tickets_cruzados.empty:
        tickets_ml = aplicar_modelo_a_tickets(tickets_cruzados)
    else:
        tickets_ml = pd.DataFrame()

    st.session_state.users = users
    st.session_state.tickets_ml = tickets_ml
    st.session_state.ruta_users = ruta_users
    st.session_state.ruta_tickets = ruta_tickets
    st.session_state.data_signature = nueva_signature


# ============================================================
# POWER BI
# ============================================================

def extraer_src_iframe(texto):
    match = re.search(r'src=["\']([^"\']+)["\']', texto, re.IGNORECASE)
    if match:
        return match.group(1)
    return texto.strip()


def normalizar_powerbi_link(texto):
    texto = texto.strip()

    if "<iframe" in texto.lower():
        return extraer_src_iframe(texto)

    return texto


def vista_power_bi():
    header()

    st.subheader("Power BI")

    st.markdown(
        """
        En esta sección puedes visualizar un reporte de Power BI conectado al proyecto.

        Puedes pegar un enlace público, un link de inserción o un iframe generado desde Power BI.
        """
    )

    modo = st.radio(
        "Modo de visualización",
        ["Dashboard simulado", "Visualizar reporte Power BI"],
        horizontal=True,
        key="modo_powerbi_app"
    )

    df = st.session_state.tickets_ml.copy()

    if modo == "Dashboard simulado":
        if df.empty:
            st.warning("No hay tickets cargados para mostrar el dashboard.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickets clasificados", len(df))
        c2.metric("Confianza promedio", f"{df['Confianza_Modelo'].mean():.2%}")
        c3.metric("Revisión humana", int((df["Requiere_Revision"] == "Sí").sum()))
        c4.metric("Automatización", f"{df['Requiere_Revision'].eq('No').mean():.2%}")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            grupo_count = df["Grupo_Asignado_ML"].value_counts().reset_index()
            grupo_count.columns = ["Grupo asignado", "Cantidad"]
            fig1 = px.bar(
                grupo_count,
                x="Grupo asignado",
                y="Cantidad",
                text="Cantidad",
                title="Tickets por grupo asignado"
            )
            fig1.update_traces(textposition="outside")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            prioridad_count = df["Prioridad_Predicha"].value_counts().reset_index()
            prioridad_count.columns = ["Prioridad", "Cantidad"]
            fig2 = px.pie(
                prioridad_count,
                names="Prioridad",
                values="Cantidad",
                title="Distribución de prioridad predicha"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        fig3 = px.sunburst(
            df,
            path=["Grupo_Asignado_ML", "Prioridad_Predicha", "Requiere_Revision"],
            title="Vista tipo Power BI: grupo, prioridad y revisión"
        )
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info(
            "Pega el link o iframe de Power BI. Si el reporte tiene permisos privados, "
            "el usuario deberá iniciar sesión con una cuenta autorizada."
        )

        powerbi_input = st.text_area(
            "Pegar link o iframe de Power BI",
            height=160,
            placeholder='Pega aquí el link o iframe. Ejemplo: <iframe title="Reporte" src="https://app.powerbi.com/reportEmbed?..."></iframe>',
            key="powerbi_input_app"
        )

        altura = st.slider(
            "Altura del reporte",
            min_value=600,
            max_value=1300,
            value=900,
            step=50,
            key="altura_powerbi_app"
        )

        if powerbi_input.strip():
            url = normalizar_powerbi_link(powerbi_input)
            st.success("Reporte cargado. Si no se visualiza, revisa que el enlace tenga permisos de visualización o sea un iframe válido.")

            st.markdown(
                f"""
                <div class="powerbi-box">
                    <iframe 
                        src="{url}"
                        width="100%"
                        height="{altura}"
                        frameborder="0"
                        allowFullScreen="true">
                    </iframe>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(f"[Abrir reporte en nueva pestaña]({url})")
        else:
            st.warning("Aún no se ha ingresado ningún link o iframe de Power BI.")


# ============================================================
# VISTAS
# ============================================================

def vista_inicio():
    header()

    df = st.session_state.tickets_ml.copy()
    users = st.session_state.users.copy()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Tickets procesados", len(df), "Desde base_tickets + nuevos tickets")

    with c2:
        if not df.empty:
            kpi_card("Confianza promedio", f"{df['Confianza_Modelo'].mean():.2%}", "Clasificación automática")
        else:
            kpi_card("Confianza promedio", "0%", "Sin tickets")

    with c3:
        if not df.empty:
            kpi_card("Revisión humana", int((df["Requiere_Revision"] == "Sí").sum()), "Casos con validación adicional")
        else:
            kpi_card("Revisión humana", "0", "Sin tickets")

    with c4:
        kpi_card("Usuarios cargados", len(users), "Desde BD_USERS")

    st.divider()

    st.subheader("Estado de carga de bases")

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.ruta_tickets:
            st.markdown(
                f"<div class='ok-box'>Base de tickets cargada:<br><b>{st.session_state.ruta_tickets}</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='warn-box'>No se encontró ni se cargó la base de tickets.</div>",
                unsafe_allow_html=True
            )

    with col2:
        if st.session_state.ruta_users:
            st.markdown(
                f"<div class='ok-box'>BD de usuarios cargada:<br><b>{st.session_state.ruta_users}</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='warn-box'>No se encontró ni se cargó la BD de usuarios.</div>",
                unsafe_allow_html=True
            )

    st.divider()

    st.subheader("Flujo de trabajo")

    st.markdown(
        """
        La aplicación trabaja únicamente con **base_tickets** y **BD_USERS**.

        **Proceso aplicado:**

        1. Se carga la base de tickets.
        2. Se carga la base de usuarios.
        3. Se cruzan ambas bases mediante la columna **Matrícula**.
        4. Se procesa el título y la descripción mediante conversión semántica.
        5. Se predice la prioridad del ticket.
        6. Se asigna automáticamente el grupo responsable:
           - **Mesa de Ayuda N1:** casos simples u operativos.
           - **Mesa de Ayuda N2:** casos técnicos o funcionales intermedios.
           - **Mesa de Ayuda N3:** casos críticos, complejos o expertos.
        7. Si el usuario marca el ticket como crítico, se deriva directamente a **Mesa de Ayuda N3**.
        8. Cualquier ticket creado en la sección **Nuevo ticket** se refleja en la bandeja y métricas.
        """
    )


def vista_bases():
    header()

    st.subheader("Bases cargadas")

    tab1, tab2 = st.tabs(["Base de tickets clasificada", "BD de usuarios"])

    with tab1:
        st.caption(f"Archivo: {st.session_state.ruta_tickets}")
        if st.session_state.tickets_ml.empty:
            st.warning("No hay tickets cargados.")
        else:
            st.dataframe(st.session_state.tickets_ml, use_container_width=True, hide_index=True)

    with tab2:
        st.caption(f"Archivo: {st.session_state.ruta_users}")
        if st.session_state.users.empty:
            st.warning("No hay usuarios cargados.")
        else:
            st.dataframe(st.session_state.users, use_container_width=True, hide_index=True)


def vista_nuevo_ticket():
    header()

    st.subheader("Registrar y clasificar nuevo ticket")

    users = st.session_state.users.copy()

    if users.empty:
        st.warning("No se encontró BD_USERS. Igual puedes registrar el ticket manualmente.")
        matriculas = ["Sin matrícula"]
    else:
        matriculas = sorted(users["Matricula"].dropna().astype(str).unique())

    with st.form("form_nuevo_ticket"):
        col1, col2 = st.columns(2)

        with col1:
            matricula = st.selectbox("Matrícula solicitante", matriculas)
            ticket_tipo = st.selectbox("Tipo de ticket", ["INC", "RITM", "CHG"])
            titulo = st.text_input("Título", placeholder="Ejemplo: Sistema no permite cargar archivo")
            estado = st.selectbox("Estado", ["Pendiente", "Abierto", "En progreso", "Cerrado", "Planificado"])
            es_critico = st.checkbox("Marcar como crítico")

        with col2:
            descripcion = st.text_area(
                "Descripción",
                height=180,
                placeholder="Describe el problema, solicitud, impacto o urgencia del ticket."
            )

        enviar = st.form_submit_button("Clasificar ticket", use_container_width=True)

    if enviar:
        if not titulo.strip() or not descripcion.strip():
            st.warning("Completa el título y la descripción.")
            return

        resultado = clasificar_ticket(
            titulo=titulo,
            descripcion=descripcion,
            estado=estado,
            es_critico=es_critico
        )

        if users.empty or matricula == "Sin matrícula":
            nombre = "No disponible"
            area = "No disponible"
            correo = "No disponible"
        else:
            usuario = users[users["Matricula"] == matricula]

            if usuario.empty:
                nombre = "No encontrado"
                area = "No encontrado"
                correo = "No encontrado"
            else:
                usuario = usuario.iloc[0]
                nombre = usuario.get("Nombre", "No disponible")
                area = usuario.get("Area", "No disponible")
                correo = usuario.get("Correo", "No disponible")

        nuevo_id = f"{ticket_tipo}{len(st.session_state.tickets_ml) + 1:07d}"

        nuevo = pd.DataFrame(
            [
                {
                    "Matricula": matricula,
                    "Ticket": nuevo_id,
                    "Prioridad": "Sin clasificar",
                    "Titulo": titulo,
                    "Descripcion": descripcion,
                    "Creado": datetime.now(),
                    "Asignado": resultado["Grupo_Asignado_ML"],
                    "Estado": estado,
                    "Nombre": nombre,
                    "Area": area,
                    "Correo": correo,
                    "Categoria_Semantica": resultado["Categoria_Semantica"],
                    "Prioridad_Predicha": resultado["Prioridad_Predicha"],
                    "Grupo_Asignado_ML": resultado["Grupo_Asignado_ML"],
                    "Nivel_Soporte": resultado["Nivel_Soporte"],
                    "Confianza_Modelo": resultado["Confianza_Modelo"],
                    "Requiere_Revision": resultado["Requiere_Revision"],
                    "Motivo_Revision": resultado["Motivo_Revision"],
                    "Score_Modelo": resultado["Score"],
                    "Marcado_Critico": resultado["Marcado_Critico"],
                    "Estado_Publicacion": "Clasificado automáticamente"
                }
            ]
        )

        st.session_state.tickets_ml = pd.concat(
            [st.session_state.tickets_ml, nuevo],
            ignore_index=True
        )

        st.success(f"Ticket {nuevo_id} registrado y clasificado correctamente. Ya aparece en la bandeja.")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Prioridad predicha", resultado["Prioridad_Predicha"])
        c2.metric("Grupo asignado", resultado["Grupo_Asignado_ML"])
        c3.metric("Confianza", f"{resultado['Confianza_Modelo']:.2%}")
        c4.metric("Requiere revisión", resultado["Requiere_Revision"])

        st.subheader("Probabilidades simuladas")

        df_probs = pd.DataFrame(
            {
                "Prioridad": list(resultado["Probabilidades"].keys()),
                "Probabilidad": list(resultado["Probabilidades"].values())
            }
        ).sort_values("Probabilidad", ascending=False)

        fig = px.bar(
            df_probs,
            x="Prioridad",
            y="Probabilidad",
            text="Probabilidad",
            title="Salida simulada del modelo"
        )

        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Variables semánticas generadas")

        variables_df = pd.DataFrame(
            [{"Variable": k, "Valor": v} for k, v in resultado["Variables"].items()]
        )

        st.dataframe(variables_df, use_container_width=True, hide_index=True)


def vista_bandeja():
    header()

    st.subheader("Bandeja de tickets clasificados")

    df = st.session_state.tickets_ml.copy()

    if df.empty:
        st.warning("No hay tickets cargados.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        filtro_prioridad = st.multiselect(
            "Prioridad predicha",
            sorted(df["Prioridad_Predicha"].dropna().unique()),
            default=sorted(df["Prioridad_Predicha"].dropna().unique())
        )

    with col2:
        filtro_grupo = st.multiselect(
            "Grupo asignado",
            sorted(df["Grupo_Asignado_ML"].dropna().unique()),
            default=sorted(df["Grupo_Asignado_ML"].dropna().unique())
        )

    with col3:
        filtro_revision = st.multiselect(
            "Requiere revisión",
            sorted(df["Requiere_Revision"].dropna().unique()),
            default=sorted(df["Requiere_Revision"].dropna().unique())
        )

    filtrado = df[
        df["Prioridad_Predicha"].isin(filtro_prioridad)
        & df["Grupo_Asignado_ML"].isin(filtro_grupo)
        & df["Requiere_Revision"].isin(filtro_revision)
    ]

    columnas_mostrar = [
        "Matricula",
        "Nombre",
        "Area",
        "Ticket",
        "Titulo",
        "Descripcion",
        "Estado",
        "Prioridad_Predicha",
        "Grupo_Asignado_ML",
        "Nivel_Soporte",
        "Categoria_Semantica",
        "Confianza_Modelo",
        "Marcado_Critico",
        "Requiere_Revision",
        "Motivo_Revision",
        "Estado_Publicacion"
    ]

    columnas_mostrar = [c for c in columnas_mostrar if c in filtrado.columns]

    st.dataframe(
        filtrado[columnas_mostrar],
        use_container_width=True,
        hide_index=True
    )

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        filtrado.to_excel(writer, index=False, sheet_name="Tickets_Clasificados")

    st.download_button(
        "Descargar clasificación en Excel",
        data=output.getvalue(),
        file_name="tickets_clasificados_ml.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )


def vista_metricas():
    header()

    st.subheader("Métricas de clasificación automática")

    df = st.session_state.tickets_ml.copy()

    if df.empty:
        st.warning("No hay datos para graficar.")
        return

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Tickets clasificados", len(df))
    c2.metric("Confianza promedio", f"{df['Confianza_Modelo'].mean():.2%}")
    c3.metric("Revisión humana", int((df["Requiere_Revision"] == "Sí").sum()))
    c4.metric("Automatización", f"{(df['Requiere_Revision'].eq('No').mean()):.2%}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        prioridad_count = df["Prioridad_Predicha"].value_counts().reset_index()
        prioridad_count.columns = ["Prioridad", "Cantidad"]

        fig1 = px.bar(
            prioridad_count,
            x="Prioridad",
            y="Cantidad",
            text="Cantidad",
            title="Distribución de prioridades predichas"
        )
        fig1.update_traces(textposition="outside")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        grupo_count = df["Grupo_Asignado_ML"].value_counts().reset_index()
        grupo_count.columns = ["Grupo asignado", "Cantidad"]

        fig2 = px.bar(
            grupo_count,
            x="Grupo asignado",
            y="Cantidad",
            text="Cantidad",
            title="Distribución por grupo asignado"
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        revision_count = df["Requiere_Revision"].value_counts().reset_index()
        revision_count.columns = ["Requiere revisión", "Cantidad"]

        fig3 = px.pie(
            revision_count,
            names="Requiere revisión",
            values="Cantidad",
            title="Tickets con revisión humana"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        categoria_count = df["Categoria_Semantica"].value_counts().reset_index()
        categoria_count.columns = ["Categoría semántica", "Cantidad"]

        fig4 = px.bar(
            categoria_count,
            x="Cantidad",
            y="Categoría semántica",
            orientation="h",
            text="Cantidad",
            title="Tickets por categoría semántica"
        )
        fig4.update_traces(textposition="outside")
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    st.subheader("Confianza por grupo asignado")

    fig5 = px.box(
        df,
        x="Grupo_Asignado_ML",
        y="Confianza_Modelo",
        points="all",
        title="Distribución de confianza por nivel de soporte"
    )

    fig5.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Resumen por grupo")

    resumen = df.groupby("Grupo_Asignado_ML").agg(
        Tickets=("Ticket", "count"),
        Confianza_Promedio=("Confianza_Modelo", "mean"),
        Requieren_Revision=("Requiere_Revision", lambda x: (x == "Sí").sum())
    ).reset_index()

    resumen["Confianza_Promedio"] = resumen["Confianza_Promedio"].round(3)

    st.dataframe(resumen, use_container_width=True, hide_index=True)


def vista_clasificacion_automatica():
    header()

    st.subheader("Clasificación automática generada")

    df = st.session_state.tickets_ml.copy()

    if df.empty:
        st.warning("No hay tickets clasificados.")
        return

    columnas = [
        "Ticket",
        "Titulo",
        "Descripcion",
        "Categoria_Semantica",
        "Prioridad_Predicha",
        "Grupo_Asignado_ML",
        "Nivel_Soporte",
        "Confianza_Modelo",
        "Marcado_Critico",
        "Requiere_Revision",
        "Motivo_Revision"
    ]

    columnas = [c for c in columnas if c in df.columns]

    st.dataframe(df[columnas], use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Interpretación")

    st.markdown(
        """
        - **Mesa de Ayuda N1:** atiende solicitudes simples, accesos básicos, cargas operativas o requerimientos frecuentes.
        - **Mesa de Ayuda N2:** atiende incidentes funcionales, validaciones, inconsistencias, actualizaciones o errores técnicos.
        - **Mesa de Ayuda N3:** atiende casos críticos, de impacto alto, producción, caídas, afectaciones masivas o temas que requieren expertos.
        """
    )


# ============================================================
# APP PRINCIPAL
# ============================================================

def main():
    with st.sidebar:
        asistente_sidebar()

        st.markdown("### 🎫 Menú principal")

        opcion = st.radio(
            "Navegación",
            [
                "Inicio",
                "Bases cargadas",
                "Clasificación automática",
                "Nuevo ticket",
                "Bandeja de tickets",
                "Métricas",
                "Power BI"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        tickets_file, users_file = cargar_archivos_sidebar()

        st.divider()

        recargar = st.button("Recargar datos", use_container_width=True)

    procesar_datos(tickets_file, users_file, forzar=recargar)

    if opcion == "Inicio":
        vista_inicio()

    elif opcion == "Bases cargadas":
        vista_bases()

    elif opcion == "Clasificación automática":
        vista_clasificacion_automatica()

    elif opcion == "Nuevo ticket":
        vista_nuevo_ticket()

    elif opcion == "Bandeja de tickets":
        vista_bandeja()

    elif opcion == "Métricas":
        vista_metricas()

    elif opcion == "Power BI":
        vista_power_bi()


if __name__ == "__main__":
    main()