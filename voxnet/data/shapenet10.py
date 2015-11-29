

class_id_to_name = {
    "1": "bathtub",
    "2": "bed",
    "3": "chair",
    "4": "desk",
    "5": "dresser",
    "6": "monitor",
    "7": "night_stand",
    "8": "sofa",
    "9": "table",
    "10": "toilet"
}
class_name_to_id = { v : k for k, v in class_id_to_name.items() }

class_names = set(class_id_to_name.values())
