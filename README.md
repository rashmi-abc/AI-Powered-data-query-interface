# AI-Powered Data Query Interface

## Introduction
The AI-Powered Query Interface is a Streamlit-based web application designed to facilitate natural language interaction with databases. The application leverages advanced AI models to generate SQL queries and provide responses based on the database schema and user queries. It includes authentication, text-to-speech (TTS) capabilities, and an option to generate entity-relationship diagrams.

## Features
- **Natural Language Processing:** Converts user queries into SQL queries using advanced AI models.
- **Text-to-Speech (TTS):** Optional TTS output for responses.
- **Authentication:** Secure access to the query interface using Firebase.
- **Database Connection:** Supports MySQL and PostgreSQL databases.
- **Entity-Relationship Diagrams:** Generates and displays ER diagrams of the connected database.
- **Logs Export:** Exports query logs for audit and analysis.
- **Voice Input:** Allows users to enter queries using voice commands via speech recognition.
- **Database Management:** Supports database creation, snapshot storage, and restoration.
- **Data Visualisation:** Supports data visualisation for greater readibility using piecharts and bargraphs.
- **Chat History:** Chat history of every user's login is visible for future use.

## System Requirements
- Python 3.8+
- Streamlit
- MySQL or PostgreSQL
- Firebase for authentication
- Environment variables for API keys and database credentials

## Query Interface
### Functionality
- Users can input natural language queries, which are converted to SQL queries using the AI model.
- The application executes the SQL query and returns the result.

### Voice Input
- Users can interact with the application using voice commands.
- This feature is implemented using the `speech_recognition` library in Python.

### TTS Output
- An optional TTS feature reads out the response.
- This can be enabled or disabled via a checkbox in the sidebar.

## Entity-Relationship Diagram Generation
### Functionality
- The "ER Diagram" tab allows users to generate an ER diagram of the connected database.
- The diagram is displayed as an image within the application.

### Implementation
- The `generate_er_diagram` function uses SQLAlchemy and Graphviz to create the ER diagram.

## Logs Export
### Functionality
- The "Export Logs" tab provides an option to download the query logs as a CSV file.
- The CSV file includes details like username, date and time, user query, and chatbot response.
- This feature helps in auditing and analyzing user queries and responses.

### Implementation
- Logs are saved in `logs/query_logs.csv` and can be downloaded via the Streamlit download button.

## Creating Database
### Functionality
- Users can create a new database directly from the application interface.
- This feature supports both MySQL and PostgreSQL databases.

### Configuration
- Database creation parameters, such as the new database name, are input via an expandable section in the Streamlit sidebar.

### Implementation
- The `create_database` function handles the creation of the database.
- It connects to the server using the provided credentials and executes SQL commands to create the database.
- It provides feedback to the user upon success or failure.

## Load Database
### Functionality
- Users can load a database schema and data from an SQL file.
- This option is useful for initializing a database with pre-defined tables and records.

### Configuration
- Users can upload an SQL file via an expandable section in the Streamlit sidebar.
- The application reads and executes the SQL commands contained in the file.

### Implementation
- The `execute_sql_file` function reads the SQL file and executes its contents on the connected database.
- It handles multi-statement execution and provides feedback on the success or failure of the operation.
- The function ensures that the database connection is properly managed and any errors are reported to the user.

## Database Snapshot
### Functionality
- The Streamlit sidebar has a feature called "Database Snapshot" along with "Create Database" and "Restore Database".
- This feature allows users to save the details of the database.
- If any details are altered by mistake, users can restore the database.

### Configuration
- Users can save snapshots of the database by entering all the database details and clicking "Create Snapshot" in the Streamlit sidebar.
- This operation downloads an `.sql` file, which can later be used to restore the database.
- The "Restore Database" option provides a dropdown listing all available `.sql` files for reloading the database.

### Implementation
- The `create_snapshot` function reads database details such as table names, column names, and values, and exports them as an `.sql` file into the project directory.
- The `restore_snapshot` function allows users to select available `.sql` files in the project directory.
- It reads the details inside the `.sql` file and restores the database to its previous state.

  

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- MySQL/PostgreSQL
- Streamlit
- Firebase SDK for Python

### Clone the Repository
```bash
 git clone https://github.com/your-username/AI-Query-Interface.git
 cd AI-Query-Interface
```

### Install Dependencies
```bash
 pip install -r requirements.txt
```

### Run the Application
```bash
 streamlit run app.py
```

## Contribution
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push to your fork and submit a Pull Request.



