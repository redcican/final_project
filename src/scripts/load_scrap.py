import base
from optirodig.models import Schrott, GiessereiSchrott, SchrottChemi
import pandas as pd
from sqlalchemy import create_engine
from django.conf import settings


def run():
    Schrott.objects.all().delete()
    
    df = pd.read_excel('D:\Python\easyml\src\scripts\scrap.xlsx')
    
    user = settings.DATABASES['default']['USER']
    password = settings.DATABASES['default']['PASSWORD']
    database_name = settings.DATABASES['default']['NAME']
    
    database_url = f"postgresql://{user}:{password}@localhost:5432/{database_name}"
    engine = create_engine(database_url)
    
    df.to_sql(Schrott._meta.db_table, engine, if_exists='append', index=False)
    
def run_giesserei():
    GiessereiSchrott.objects.all().delete()
    
    df = pd.read_excel('D:\Python\easyml\src\scripts\giesserei.xlsx')
    
    user = settings.DATABASES['default']['USER']
    password = settings.DATABASES['default']['PASSWORD']
    database_name = settings.DATABASES['default']['NAME']
    
    database_url = f"postgresql://{user}:{password}@localhost:5432/{database_name}"
    engine = create_engine(database_url)
    
    df.to_sql(GiessereiSchrott._meta.db_table, engine, if_exists='append', index=False)
    
def run_chemi():
    SchrottChemi.objects.all().delete()
    
    df = pd.read_excel('D:\Python\easyml\src\scripts\chemi.xlsx')
    
    user = settings.DATABASES['default']['USER']
    password = settings.DATABASES['default']['PASSWORD']
    database_name = settings.DATABASES['default']['NAME']
    
    database_url = f"postgresql://{user}:{password}@localhost:5432/{database_name}"
    engine = create_engine(database_url)
    
    df.to_sql(SchrottChemi._meta.db_table, engine, if_exists='append', index=False)
    
if __name__ == '__main__':
    #run()
    #run_giesserei()
    run_chemi()