import json

import requests

title = "Perceuse sans fil RYOBI ONE PLUS 18 V 4 Ah, 2 batteries RPD18-2C42TA55"
description = (
    "Gamme ONE+, une batterie compatible avec de nombreux outils de la gamme en outillage et jardin "
    "Perceuse-visseuse à percussion 18V ONE+™ idéale pour percer le bois ou le métal, visser ou même "
    " percer la maçonnerie grâce à la fonction de percussion. "
)
attribute_code = "99999"

print(json.dumps({"title": title, "description": description, "attribute_code": attribute_code}))


response = requests.post(
    "http://127.0.0.1:3000/extract",
    headers={"content-type": "application/json"},
    json={"instances": [{"title": title, "description": description, "attribute_code": attribute_code}]},
)


print("This is the response")
print(response.content)


# print("===================================")

# attribute_codes = ["02419","15344", "01746", "00562", "99999"]

# for att_code in attribute_codes:
#     print(att_code)
#     response = requests.post(
#         "http://127.0.0.1:3000/extract",
#         headers={"content-type": "application/json"},
#         json={"instances": [{"title": title, "description": description, "attribute_code": att_code}]},
#     )
    
#     print(response.content)
    