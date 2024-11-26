# %%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import time
import os

# List of authors to fill in the text box

philosophers = [
    # "Plato",
    # "Aristotle",
    # "Socrates",
    "Pythagoras",
    "Ptolemy",
    "Euclid or Archimedes",
    "Diogenes",
    "Zeno of Citium",
    "Heraclitus",
    "Epicurus",
    "Alcibiades",
    "Averroes",
    "Anaximander",
    "Parmenides",
    "Democritus",
    "Plotinus",
    "Hypatia",
    "Apollodorus"
]

# %%

# Initialize WebDriver (make sure ChromeDriver is installed and path is set)
driver = webdriver.Chrome()

try:
    # Navigate to the webpage
    
    # Fill the text box with authors and click on list elements
    for author in philosophers:

        driver.get('https://www.gutenberg.org/')
        # Fill the text input field
        text_box_xpath = "/html/body/div[1]/nav/div[2]/div/form/input[1]"
        text_box = driver.find_element(By.XPATH, text_box_xpath)
        text_box.clear()  # Clear any existing text
        text_box.send_keys(author)  # Input author name
        text_box.send_keys(Keys.RETURN) 
        time.sleep(0.5)  # Short delay to simulate typing

        author_search_xpath ="/html/body/div[2]/div[2]/div/ul/li[1]/a/span[2]/span[1]"
        element = driver.find_element(By.XPATH, author_search_xpath)
        element.click()
        time.sleep(0.5) 

        first_author_xpath ="/html/body/div[2]/div[2]/div/ul/li[2]/a/span[2]/span[1]"
        # Start clicking the list items
        element = driver.find_element(By.XPATH, first_author_xpath)
        element.click()
        time.sleep(0.5) 

        i = 6 # book list starts here

        while True:
            xpath_book = f"/html/body/div[2]/div[2]/div/ul/li[{i}]"
            xpath_book_title = f"/html/body/div[2]/div[2]/div/ul/li[{i}]/a/span[2]/span[1]"

            try:
                # Get the book title
                title_element = driver.find_element(By.XPATH, xpath_book_title)
                book_title = title_element.text

                # Find the book element and click it
                element = driver.find_element(By.XPATH, xpath_book)
                element.click()
                time.sleep(1)  # Wait for a second

                # Click for UTF-8 version
                element = driver.find_element(By.PARTIAL_LINK_TEXT, "Text UTF-8")
                element.click()
                time.sleep(1)

                # Get the entire HTML source
                page_source = driver.page_source

                # Save the HTML content to a .txt file
                with open(f'data/philosophy_data/raw/{author}/{book_title}_content.txt', 'w', encoding='utf-8') as file:
                    file.write(page_source)

                print(f"Page content has been saved as '{author}_{book_title}_content.txt'.")

                i += 1  # Move to the next book in the list

                driver.back()
                driver.back()
            except :
                # Break the loop if there are no more elements to click
                break
        
        time.sleep(1)  # Wait before moving to the next author

finally:
    # Close the driver after finishing
    driver.quit()
# %%



# %%

def make_directories(authors):
    # Create directories for each author in the list
    base_dir = 'data/philosophy_data/raw'
    os.makedirs(base_dir, exist_ok=True)

    for author in authors:
        author_dir = os.path.join(base_dir, author)
        os.makedirs(author_dir, exist_ok=True)

make_directories(philosophers)