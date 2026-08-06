#List, Dictionary
'''
# characters= ["Harry", "Ron", "Hermione", "Snape"]
# for characters in characters:
#   print(characters) 

# characters= ["Harry Potter", "Ron Weasley", "Hermione Granger"]
# print("first character: " , characters[0])
# characters.append("Albus Dumbledore")
# print("Total Characters :", len(characters))
# for characters in characters:
#   print(characters) 

# character = { "name": "Harry Potter", "house": "Gryffindor", "mentions": 356 }
# print("Character Name : ", character["name"])
# print("House : ", character["house"])
# print("Mentions : ", character["mentions"])
'''

#Function
'''
# def greet():
#   print("Welcome to LitMind")
# greet()

# def show_book(book_name):
#   print("Book: ", book_name)
# show_book("Harry Potter")

# def count_characters(characters):
#     print(len(characters))
# count_characters(["Harry", "Ron", "Hermione"])
'''

#File Handling
'''
# with open("practice/sample.txt", "r") as file:
#   content= file.read()

# print(content)

# with open("practice/sample.txt", "a") as file:
#   file.write("\nAI Powered Novel Analyzer")

# with open("practice/sample.txt", "w") as file:
#   file.write("Hello LitMind")
'''

#Pathlib
'''
# from pathlib import Path

# file_path= Path("practice/sample.txt")
# print(file_path)

# with open(file_path, "r") as file:
#   content= file.read()

# print(content)
'''

#Exceptional Handling
from pathlib import Path

file_path = Path("practice/sample.txt")

try:
  with open(file_path, "r") as file:
    content = file.read()

  print(content)

except FileNotFoundError:
  print("File not found.")