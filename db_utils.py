# db_utils.py
from sqlalchemy import create_engine

def get_active_engine(session=None):
    """Retourne l'engine pour le dataset actif"""
    if session and session.get('active_dataset', 'database') != 'database':
        # Logique pour dataset personnalisé si nécessaire
        pass

    return create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")



'''
def get_active_engine():
    """Retourne l'engine pour le dataset actif"""
    active = session.get('active_dataset', 'database')

    if active == 'database':
        return create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")
    else:
        # Le dataset a déjà été chargé dans la base lors de l'activation
        return create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")
        
        
'''