from api_test.app.db import Base, engine

if __name__ == "__main__":
    print("Création des tables...")
    Base.metadata.create_all(bind=engine)
    print("Terminé.") 