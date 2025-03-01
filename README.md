**AI-Powered Data Query Interface**

**Introduction **
The AI Powered Query Interface is a Streamlit-based web application designed 
to facilitate natural language interaction with databases. The application 
leverages advanced AI models for generating SQL queries and providing 
responses based on the database schema and user queries. It includes 
authentication, text-to-speech (TTS) capabilities, and an option to generate 
entity-relationship diagrams. 

**Features **
● Natural Language Processing: Convert user queries into SQL queries 
using advanced AI models.   
● Text-to-Speech (TTS): Optional TTS output for responses.   
● Authentication: Secure access to the query interface.   
● Database Connection: Supports MySQL and PostgreSQL databases.   
● Entity-Relationship Diagrams: Generate and display ER diagrams of 
the connected database.   
● Logs Export: Export query logs for audit and analysis. 

**System Requirements **
● Python 3.8+   
● Streamlit   
● MySQL or PostgreSQL   
● Environment variables for API keys and database credentials   
● Firebase  

**Proposed Solution **
● Creation of an AI Powered Query Interface is a Python Streamlit application 
that enables natural language interactions with MySQL and PostgreSQL 
databases.   
● It uses advanced AI models to convert user queries into SQL queries, returning 
results in natural language.   
● The application supports text-to-speech (TTS) output for responses and 
includes user authentication for secure access.   
● Users can visualize the database schema through auto-generated 
entity-relationship (ER) diagrams.   
● Query logs are maintained and can be exported for audit purposes.   
● The system integrates various libraries, including SQLAlchemy, Graphviz, and 
pyttsx3, to provide a seamless and interactive database querying experience.   
● For an authentication purpose of the application, it is integrated with the 
Firebase so that only database administrators who have access can log into the 
application


**Implementation **
The application includes user authentication using firebase so that only 
registered users can use the application and all the users data can be viewed 
and managed in the firebase console. 
Configuration 
Login using the credentials or use register button to add a new user into the 
firebase for accessing the web application. 
For instance, use this login credentials 
USERNAME:admin@gmail.com 
PASSWORD:admin123 

**Database Connection **
Supported Databases 
● MySQL   
● PostgreSQL 
Configuration   
Database connection parameters are set via the Streamlit sidebar. Users can 
input host, port, username, password, and database name to connect according 
to their server credentials
