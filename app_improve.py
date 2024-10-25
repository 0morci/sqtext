import streamlit as st
import sqlite3
import pandas as pd
from langchain.chat_models import ChatOpenAI
from sqlalchemy import create_engine, inspect
from langchain import PromptTemplate
from langchain.chains import LLMChain
from openai import OpenAI
import datetime
import os

current_time = datetime.datetime.now().date()
current_year= current_time.strftime("%Y")
current_month= current_time.strftime("%m")
current_day= current_time.strftime("%d")
current_hour= current_time.strftime("%H")
current_minute= current_time.strftime("%M")

st.title("SQText")
st.write("""Transform your questions into queries!""")

# Function to check if the provided API key is valid
def is_api_key_valid(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    # Making a simple API call to the OpenAI endpoint
    try:
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error occurred while validating API key: {e}")
        return False


# Input field for user to enter their OpenAI API key
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = None

user_api_key = st.text_input("Enter your OpenAI API key:", type="password")


# Validate the API key upon entry
if user_api_key:
    if is_api_key_valid(user_api_key):
        st.session_state.user_api_key = user_api_key
        st.success("API key is valid!")
    else:
        st.error("Invalid API key. Please check and try again.")

# Ensure user has entered their API key
if not st.session_state.user_api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()


# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.session_state.user_api_key)
db_path=''

with st.expander("Upload your DB"):
    uploaded_file = st.file_uploader('', type=['.sql', '.db', '.sqlite', '.sqlite3', '.db3'],
                                    accept_multiple_files=False)
if uploaded_file is not None:
        # Save the uploaded file to the server
        db_path = os.path.join("uploaded_database.db")  # Store in the current working directory
        with open(db_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("DB successfully uploaded!")


while uploaded_file is None:
    st.stop()


db_connection = sqlite3.connect(db_path)
cur = db_connection.cursor()

def execute_query(query):
    return pd.read_sql_query(query, db_connection)




# Conectar a la base de datos (ajusta la cadena de conexión según tu BBDD)
engine = create_engine(f"""sqlite:///{db_path}""")  # Reemplaza con tu base de datos

# Inspeccionar la base de datos para obtener el esquema
inspector = inspect(engine)

# Extraer la información del esquema: tablas, columnas y tipos de datos
schema_description = {}

# Obtener las tablas de la base de datos
tables = inspector.get_table_names()

for table in tables:
    columns = inspector.get_columns(table)
    schema_description[table] = []
    
    # Para cada columna, almacenar el nombre y tipo de datos
    for column in columns:
        column_name = column['name']
        column_type = str(column['type'])
        schema_description[table].append((column_name, column_type))

# Ahora, construir el texto del esquema como un string (variable 'schema_text')
schema_text = ""
for table, columns in schema_description.items():
    schema_text += f"Table: {table}\n"
    for column_name, column_type in columns:
        schema_text += f"  - Column: {column_name}, Type: {column_type}\n"
table_description=f"This is the schema of my database:\n\n{schema_text}"


with st.expander("Check DB schema"):
    st.text(schema_text)


template_question_to_sql = """[INST] You are given data tables in SQLite.
When generating queries, always refer to the following database schema:
[DATA_DESCRIPTION]
{data_description}
[\DATA_DESCRIPTION]
Always prioritize accuracy and relevance in your responses. Your task is to generate a valid SQL query for SQLite that will retrieve the data required to answer the user's question.

Follow these guidelines:
1. Base your responses solely on the provided schema.
2. Use ONLY the tables and columns specified in the database schema.
3. Do not introduce additional terms, aliases, or assumptions not present in the schema.
4. When joining multiple tables, use table identifiers consistently.
5. Maintain original column names throughout the query.
6. If the question cannot be answered using the available tables, clearly state this limitation.

Construct your answer as a JSON object with a single key "sql_query" containing the valid SQL query:
{{"sql_query": "YOUR_SQL_QUERY_HERE"}}

Additional requirements:
- Use table aliases when joining tables (e.g., 't1', 't2') to improve readability.
- Implement appropriate JOINs (INNER, LEFT, RIGHT) based on the relationship between tables.
- Utilize WHERE clauses to filter data effectively.
- Apply aggregation functions (SUM, AVG, COUNT, etc.) when necessary.
- Include ORDER BY and LIMIT clauses if relevant to the question.
- Handle date/time operations correctly, using appropriate SQLite functions.

Remember to thoroughly analyze the user's question and the schema before formulating your query. Ensure that your query is optimized and retrieves only the necessary data to answer the question accurately.
[EXAMPLE]For example:
User question: "What is the total freight value of health_beauty items in 2017?"
Expected answer in JSON: {{"sql_query": "SELECT SUM(o.freight_value)
                                         FROM order_items as o
                                         LEFT JOIN products as p on p.product_id = o.product_id
                                         LEFT JOIN product_category_name_translation as c on c.product_category_name = p.product_category_name
                                         WHERE shipping_limit_date >= '2017-01-01'
                                            AND shipping_limit_date < '2018-01-01'
                                            AND product_category_name_english='health_beauty'"}}
                                            
You are an intelligent assistant for generating queries and providing information based on a given context. When you encounter terms that may have abbreviations or nicknames, please replace them with their full names or correct terminology. For example:
- If you see 'NY,' use 'New York.'
- If you see 'LA,' use 'Los Angeles.'
- If you encounter 'Monza,' replace it with 'Autodromo Nazionale di Monza.'
- If you find 'Monaco,' use 'Circuit de Monaco.'
- If you come across any acronyms or shortened terms, ensure to provide their complete forms.
Additionally, when generating queries or responses, if you encounter any incomplete columns or missing information, consult the provided database schema or context to find the correct identifiers and ensure your answers are accurate and complete.

You will not use the 'now' variable for the current_time, instead YOU MUST USE 'current_time' for dates.
You also must use:
- '%Y' will be 'current_year'
- '%m' will be 'current_month'
- '%d' will be 'current_day'
- hour will be current_hour
- minute will be current_minute

Remember, providing accurate and reliable information is paramount. Double-check your calculations and data sources when responding.

[/EXAMPLE]
User question: {text}
[/INST]"""


template_sql_to_insight = """[INST] You are a task answering user questions ONLY based on the provided data frame and an SQL query used to generate the table.
[EXAMPLE]For example:
User question: "What is the total freight value of health_beauty items in 2017?"
SQL query: "SELECT SUM(o.freight_value)
            FROM order_items as o
            LEFT JOIN products as p on p.product_id = o.product_id
            LEFT JOIN product_category_name_translation as c on c.product_category_name = p.product_category_name
            WHERE shipping_limit_date >= '2017-01-01'
                AND shipping_limit_date < '2018-01-01'
                AND product_category_name_english='health_beauty';"
Data frame: "	SUM(freight_value)
            0   67269.31"
Answer: "The total freight value of 'health beauty' products ordered in 2017 is 67269.31."
[/EXAMPLE]
Answer should be specific and precise, don't add anything else!
If you can't answer the question based on the provided data, say so, don't try to guess!



If the response is an empty dataframe, create a response based on that.
For example: "When was Carlos Sainz first win?"
Your response should be: "Carlos Sainz has not won any race yet."
The response should be based on the question according to an empty dataframe.

If the response is too long and you feel the need to use "etc.", don't do it. I want the whole list.

User question: {text}
SQL query: {sql_query}
Data frame: {table}
[/INST]"""

def conversation(messages, question, max_questions=10):
    prompt = "The conversation so far:\n"
    last_messages = messages[-10:]  # Slices the last 10 messages
    for msg in last_messages:
        prompt += f"{msg['role']}: {msg['content']}\n"
    full_prompt = f"""{prompt} \n You are a knowledgeable assistant capable of remembering previous interactions. When answering questions, please use the information provided in earlier messages. For example, if the user mentions a specific pilot, race, etc. make sure to incorporate that context into your answers.

Here is the question: {question}.
              """
    return prompt

def text2sql(question,
             llm_model = llm,
             question_to_sql_prompt=template_question_to_sql,
             df_to_insight_prompt = template_sql_to_insight,
             data_description = table_description):
    import json

    # Get data frame:
    prompt_sql = PromptTemplate(template=conversation(st.session_state.messages, question)+"\n"+question_to_sql_prompt, input_variables=["text", "data_description"])
    llm_chain = LLMChain(prompt=prompt_sql, llm=llm_model)

    llm_reply = llm_chain.predict(text = question, data_description = data_description)
    json_reply = json.loads(llm_reply.replace('\n',' '))
    sql_query = json_reply['sql_query']
    df_reply = execute_query(sql_query)

    # Get insight
    prompt_insight = PromptTemplate(template=df_to_insight_prompt, input_variables=["text", "sql_query", "table"])
    llm_chain = LLMChain(prompt=prompt_insight, llm=llm_model)
    llm_reply = llm_chain.predict(text = question, sql_query = sql_query, table = df_reply)
    response= f""" {llm_reply} \n\n {sql_query}"""
    return response

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt_chat := st.chat_input("Ask me anything about your database!"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt_chat})
    with st.chat_message("user"):
        st.markdown(prompt_chat)
        

    stream = text2sql(prompt_chat)            

        
    with st.chat_message("assistant"):
        # Asegúrate de que `stream` sea un objeto iterable o de flujo
        response = ''.join(stream)  # Unimos las partes del stream en una sola cadena
        st.write(response)  # Imprimimos la respuesta en la app

    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    st.session_state.messages.append({"role": "assistant", "content": response})
