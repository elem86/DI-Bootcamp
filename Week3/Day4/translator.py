from deep_translator import GoogleTranslator

french_words = ["Bonjour", "Au revoir", "Bienvenue", "A bientôt"]

translator = GoogleTranslator(source="auto", target="en")
translations = {word: translator.translate(word) for word in french_words}

print(translations)
