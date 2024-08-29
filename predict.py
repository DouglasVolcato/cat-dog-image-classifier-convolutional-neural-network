from Utils.ImageClassifierModel import ImageClassifierModel


model = ImageClassifierModel()

print(f"Should be DOG: {model.predict('cache/test/17.jpg')}")
print(f"Should be DOG: {model.predict('cache/test/43.jpg')}")
print(f"Should be DOG: {model.predict('cache/test/27.jpg')}")
print(f"Should be CAT: {model.predict('cache/test/40.jpg')}")
print(f"Should be CAT: {model.predict('cache/test/11.jpg')}")
print(f"Should be CAT: {model.predict('cache/test/6.jpg')}")
