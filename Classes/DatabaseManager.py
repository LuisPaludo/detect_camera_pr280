import psycopg2
import os
import logging

class DatabaseManager:
    """
    Classe para gerenciar a conexão e operações com o banco de dados PostgreSQL.
    """

    def __init__(self):
        """
        Inicializa o gerenciador de banco de dados com as configurações do ambiente.
        Obterá as configurações de conexão a partir de variáveis de ambiente.
        """
        self.connection = None
        self.cursor = None

        # Configurações de conexão (recuperadas de variáveis de ambiente)
        self.db_config = {
            'host': os.environ.get('DB_HOST', 'postgres'),
            'port': os.environ.get('DB_PORT', '5432'),
            'database': os.environ.get('DB_NAME', 'camera_detection'),
            'user': os.environ.get('DB_USER', 'postgres'),
            'password': os.environ.get('DB_PASSWORD', 'postgres')
        }

        self.logger = logging.getLogger(__name__)
        self.is_saving_enabled = False  # Flag para habilitar/desabilitar o salvamento

        # Inicializa a conexão com o banco de dados
        self._init_database()

    def _init_database(self):
        """
        Inicializa a conexão com o banco de dados e cria tabelas necessárias se não existirem.
        """
        try:
            # Conectar ao banco de dados
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()

            # Criar tabelas se não existirem
            self._create_tables()

            self.logger.info("Conexão com o banco de dados estabelecida com sucesso")

        except Exception as e:
            self.logger.error(f"Erro ao conectar ao banco de dados: {str(e)}")
            # Não levanta exceção para permitir que o programa continue mesmo sem BD

    def close(self):
        """
        Fecha a conexão com o banco de dados.
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            self.logger.info("Conexão com o banco de dados fechada")