import os, sys, sqlite3



connection = sqlite3.connect("face.db")

cursor = connection.cursor()

def create_table():
  sql = "CREATE TABLE IF NOT EXISTS people(person_id INTEGER PRIMARY KEY, name TEXT, mydate TEXT, mytime TEXT)" 
  cursor.execute(sql)


create_table()

cursor.close()
connection.close()


