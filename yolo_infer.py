import json
import os
import re
import sys
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / 'models' / 'food-yolo.pt'
FALLBACK_MODEL_PATH = ROOT / 'yolov8n.pt'
NUTRITION_PATH = ROOT / 'food_nutrition.json'
IGNORED_CLASSES = {'background', 'otheringredients'}
COMBO_RULES = [
    {
        'keys': {'rajmacurry', 'whiterice'},
        'name': 'Rajma Chawal',
        'summary': 'Detected rajma curry with rice and combined it as one meal.'
    },
    {
        'keys': {'dal', 'whiterice'},
        'name': 'Dal Chawal',
        'summary': 'Detected dal with rice and combined it as one meal.'
    },
    {
        'keys': {'chole', 'bhatura'},
        'name': 'Chole Bhature',
        'summary': 'Detected chole with bhatura and combined it as one meal.'
    }
]
COMPLETE_DISH_SUPPRESSIONS = {
    'dosa': {
        'coconutchutney', 'greenchutney', 'sauce', 'onion', 'potato'
    },
    'hamburg': {
        'bread', 'lettuce', 'tomato', 'cheesebutter', 'sauce', 'onion', 'cucumber',
        'friedmeat', 'steak', 'pork', 'chickenduck'
    },
    'pizza': {
        'cheesebutter', 'tomato', 'pepper', 'onion', 'sausage', 'sauce', 'mushroom',
        'whitebuttonmushroom', 'pork', 'chickenduck', 'bread'
    },
    'biryani': {'rice', 'whiterice', 'chickenduck', 'lamb', 'muttoncurry'},
    'samosa': {'potato', 'onion', 'greenchutney', 'sauce'},
    'idli': {'coconutchutney', 'greenchutney', 'sauce'},
    'sandwich': {
        'bread', 'cheesebutter', 'tomato', 'lettuce', 'cucumber', 'pork', 'steak',
        'friedmeat', 'chickenduck', 'sausage', 'sauce'
    },
    'pasta': {
        'noodles', 'sauce', 'tomato', 'cheesebutter', 'pork', 'steak', 'friedmeat',
        'chickenduck', 'mushroom', 'whitebuttonmushroom'
    },
    'noodles': {
        'noodles', 'sauce', 'egg', 'pork', 'steak', 'friedmeat', 'chickenduck',
        'vegetable', 'springonion', 'onion', 'cabbage'
    },
    'salad': {
        'lettuce', 'tomato', 'cucumber', 'cabbage', 'carrot', 'olives', 'cheesebutter',
        'sauce', 'egg', 'onion'
    },
    'soup': {
        'sauce', 'soup', 'tomato', 'egg', 'pork', 'chickenduck', 'mushroom',
        'whitebuttonmushroom', 'kelp', 'seaweed'
    },
    'hanamakibaozi': {'bread', 'pork', 'steak', 'friedmeat', 'chickenduck'},
    'wontondumplings': {'bread', 'pork', 'steak', 'friedmeat', 'chickenduck', 'shrimp', 'crab'},
    'pie': {'bread', 'apple', 'pork', 'steak', 'friedmeat', 'chickenduck'}
}


DEFAULT_NUTRITION = {
    'dal': {'weight_g': 180, 'calories': 210, 'protein_g': 12, 'carbs_g': 28, 'fat_g': 5, 'fiber_g': 7, 'healthy': True},
    'roti': {'weight_g': 45, 'calories': 120, 'protein_g': 4, 'carbs_g': 22, 'fat_g': 3, 'fiber_g': 3, 'healthy': True},
    'rice': {'weight_g': 150, 'calories': 195, 'protein_g': 4, 'carbs_g': 42, 'fat_g': 0.4, 'fiber_g': 1, 'healthy': True},
    'burger': {'weight_g': 220, 'calories': 520, 'protein_g': 24, 'carbs_g': 45, 'fat_g': 28, 'fiber_g': 3, 'healthy': False},
    'hamburg': {'weight_g': 220, 'calories': 520, 'protein_g': 24, 'carbs_g': 45, 'fat_g': 28, 'fiber_g': 3, 'healthy': False},
    'pizza': {'weight_g': 150, 'calories': 360, 'protein_g': 15, 'carbs_g': 40, 'fat_g': 15, 'fiber_g': 2, 'healthy': False},
    'sandwich': {'weight_g': 250, 'calories': 500, 'protein_g': 22, 'carbs_g': 45, 'fat_g': 25, 'fiber_g': 4, 'healthy': False},
    'banana': {'weight_g': 118, 'calories': 105, 'protein_g': 1.3, 'carbs_g': 27, 'fat_g': 0.3, 'fiber_g': 3.1, 'healthy': True},
    'apple': {'weight_g': 182, 'calories': 95, 'protein_g': 0.5, 'carbs_g': 25, 'fat_g': 0.3, 'fiber_g': 4.4, 'healthy': True},
    'cake': {'weight_g': 85, 'calories': 280, 'protein_g': 3, 'carbs_g': 35, 'fat_g': 15, 'fiber_g': 1, 'healthy': False}
}

CATEGORY_NUTRITION = {
    'fruit': {'weight_g': 150, 'calories': 85, 'protein_g': 1, 'carbs_g': 22, 'fat_g': 0.3, 'fiber_g': 3.5, 'healthy': True},
    'vegetable': {'weight_g': 120, 'calories': 45, 'protein_g': 2, 'carbs_g': 9, 'fat_g': 0.4, 'fiber_g': 3.5, 'healthy': True},
    'nut': {'weight_g': 30, 'calories': 180, 'protein_g': 6, 'carbs_g': 6, 'fat_g': 16, 'fiber_g': 3, 'healthy': True},
    'dessert': {'weight_g': 90, 'calories': 280, 'protein_g': 4, 'carbs_g': 38, 'fat_g': 13, 'fiber_g': 1, 'healthy': False},
    'drink': {'weight_g': 220, 'calories': 130, 'protein_g': 5, 'carbs_g': 20, 'fat_g': 4, 'fiber_g': 0, 'healthy': False},
    'grain': {'weight_g': 160, 'calories': 230, 'protein_g': 6, 'carbs_g': 45, 'fat_g': 3, 'fiber_g': 3, 'healthy': True},
    'protein': {'weight_g': 150, 'calories': 300, 'protein_g': 25, 'carbs_g': 4, 'fat_g': 20, 'fiber_g': 1, 'healthy': False},
    'seafood': {'weight_g': 140, 'calories': 190, 'protein_g': 25, 'carbs_g': 2, 'fat_g': 8, 'fiber_g': 0, 'healthy': True},
    'mushroom': {'weight_g': 100, 'calories': 35, 'protein_g': 3, 'carbs_g': 6, 'fat_g': 0.4, 'fiber_g': 2.5, 'healthy': True},
    'condiment': {'weight_g': 35, 'calories': 60, 'protein_g': 1, 'carbs_g': 6, 'fat_g': 4, 'fiber_g': 1, 'healthy': False}
}

CATEGORY_CLASSES = {
    'fruit': {
        'apple', 'date', 'apricot', 'avocado', 'banana', 'strawberry', 'cherry', 'blueberry',
        'raspberry', 'mango', 'olives', 'peach', 'lemon', 'pear', 'fig', 'pineapple',
        'grape', 'kiwi', 'melon', 'orange', 'watermelon'
    },
    'vegetable': {
        'eggplant', 'potato', 'garlic', 'cauliflower', 'tomato', 'kelp', 'seaweed',
        'springonion', 'rape', 'ginger', 'okra', 'lettuce', 'pumpkin', 'cucumber',
        'whiteradish', 'carrot', 'asparagus', 'bambooshoots', 'broccoli', 'celerystick',
        'cilantromint', 'snowpeas', 'cabbage', 'beansprouts', 'onion', 'pepper',
        'greenbeans', 'frenchbeans', 'corn'
    },
    'nut': {'almond', 'redbeans', 'cashew', 'driedcranberries', 'soy', 'walnut', 'peanut'},
    'dessert': {
        'candy', 'eggtart', 'chocolate', 'biscuit', 'popcorn', 'pudding', 'icecream',
        'cake', 'pie'
    },
    'drink': {'wine', 'milkshake', 'coffee', 'juice', 'milk', 'tea'},
    'grain': {'bread', 'hanamakibaozi', 'wontondumplings', 'pasta', 'noodles', 'rice'},
    'protein': {'steak', 'pork', 'chickenduck', 'sausage', 'friedmeat', 'lamb', 'egg', 'tofu', 'hamburg', 'pizza'},
    'seafood': {'crab', 'fish', 'shellfish', 'shrimp'},
    'mushroom': {'kingoystermushroom', 'shiitake', 'enokimushroom', 'oystermushroom', 'whitebuttonmushroom'},
    'condiment': {'sauce', 'cheesebutter', 'salad', 'soup'}
}


def load_nutrition():
    nutrition = {canonical_key(key): value for key, value in DEFAULT_NUTRITION.items()}

    if not NUTRITION_PATH.exists():
        return nutrition

    with NUTRITION_PATH.open('r', encoding='utf-8') as file:
        data = json.load(file)

    nutrition.update({canonical_key(key): value for key, value in data.items()})
    return nutrition


def canonical_key(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())


def display_name(value):
    if canonical_key(value) == 'hamburg':
        return 'Burger'

    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', value.replace('_', ' '))
    return spaced.title()


def fallback_nutrition(food_name):
    key = canonical_key(food_name)
    for category, classes in CATEGORY_CLASSES.items():
        if key in classes:
            return CATEGORY_NUTRITION[category]

    return {'weight_g': 150, 'calories': 250, 'protein_g': 8, 'carbs_g': 30, 'fat_g': 9, 'fiber_g': 2, 'healthy': False}


def safe_number(value):
    if value == int(value):
        return int(value)
    return round(value, 1)


def sum_foods(name, foods, summary):
    confidence_values = [item['confidence'] for item in foods if isinstance(item.get('confidence'), (int, float))]
    healthy_count = sum(1 for item in foods if item.get('healthy'))
    combined = {
        'name': name,
        'estimated_weight_g': safe_number(sum(item['estimated_weight_g'] for item in foods)),
        'calories': safe_number(sum(item['calories'] for item in foods)),
        'protein_g': safe_number(sum(item['protein_g'] for item in foods)),
        'carbs_g': safe_number(sum(item['carbs_g'] for item in foods)),
        'fat_g': safe_number(sum(item['fat_g'] for item in foods)),
        'fiber_g': safe_number(sum(item['fiber_g'] for item in foods)),
        'confidence': safe_number(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.5,
        'healthy': healthy_count >= len(foods) / 2,
        'source_keys': [item['source_key'] for item in foods],
        'composition_note': summary
    }
    return combined


def postprocess_foods(foods):
    if not foods:
        return foods, None

    combo_notes = []
    consumed_indices = set()
    processed_dishes = []

    # PASS 1: Process all EXPLICITLY detected parent dishes first.
    # This prevents their ingredients from being stolen by other implicit dishes.
    for dish_key, ingredients in COMPLETE_DISH_SUPPRESSIONS.items():
        dish_index = None
        for i, item in enumerate(foods):
            if item['source_key'] == dish_key and i not in consumed_indices:
                dish_index = i
                break

        if dish_index is not None:
            constituent_indices = [dish_index]
            # Suppress all its ingredient detections to prevent double counting
            for i, item in enumerate(foods):
                if i != dish_index and i not in consumed_indices and item['source_key'] in ingredients:
                    consumed_indices.add(i)
            
            dish_name = display_name(dish_key)
            summary = f"Detected {dish_name} as a whole meal and suppressed duplicate ingredient detections."
            group_foods = [foods[idx] for idx in constituent_indices]
            for idx in constituent_indices:
                consumed_indices.add(idx)

            combined = sum_foods(dish_name, group_foods, summary)
            combined['source_key'] = dish_key
            processed_dishes.append(combined)
            combo_notes.append(summary)

    # PASS 2: Process IMPLICIT detections from the remaining, unconsumed ingredients.
    for dish_key, ingredients in COMPLETE_DISH_SUPPRESSIONS.items():
        # Skip if this dish was already explicitly handled in processed_dishes
        if any(d['source_key'] == dish_key for d in processed_dishes):
            continue

        implicit_detected = False
        constituent_indices = []

        for i, item in enumerate(foods):
            if i not in consumed_indices and item['source_key'] in ingredients:
                constituent_indices.append(i)

        # Check if constituents are sufficient to infer the parent dish
        if dish_key == 'pizza':
            has_base = any(foods[idx]['source_key'] in {'bread', 'cheesebutter'} for idx in constituent_indices)
            if has_base and len(constituent_indices) >= 2:
                implicit_detected = True
        elif dish_key == 'hamburg':
            has_bread = any(foods[idx]['source_key'] == 'bread' for idx in constituent_indices)
            has_meat = any(foods[idx]['source_key'] in {'friedmeat', 'steak', 'pork', 'chickenduck'} for idx in constituent_indices)
            if has_bread and has_meat:
                implicit_detected = True
        elif dish_key == 'dosa':
            has_chutney = any(foods[idx]['source_key'] in {'coconutchutney', 'greenchutney'} for idx in constituent_indices)
            has_filling = any(foods[idx]['source_key'] in {'potato', 'onion'} for idx in constituent_indices)
            if has_chutney and has_filling:
                implicit_detected = True
        elif dish_key == 'sandwich':
            has_bread = any(foods[idx]['source_key'] == 'bread' for idx in constituent_indices)
            has_filling = any(foods[idx]['source_key'] in {'cheesebutter', 'tomato', 'lettuce', 'cucumber', 'pork', 'steak', 'friedmeat', 'chickenduck', 'sausage'} for idx in constituent_indices)
            if has_bread and has_filling:
                implicit_detected = True
        elif dish_key == 'pasta':
            has_pasta = any(foods[idx]['source_key'] == 'noodles' for idx in constituent_indices)
            has_sauce = any(foods[idx]['source_key'] in {'sauce', 'tomato', 'cheesebutter'} for idx in constituent_indices)
            if has_pasta and has_sauce:
                implicit_detected = True
        elif dish_key == 'salad':
            if len(constituent_indices) >= 3:
                implicit_detected = True
        elif dish_key == 'soup':
            has_soup = any(foods[idx]['source_key'] in {'soup', 'sauce'} for idx in constituent_indices)
            has_content = any(foods[idx]['source_key'] in {'tomato', 'egg', 'pork', 'chickenduck', 'mushroom', 'whitebuttonmushroom', 'kelp', 'seaweed'} for idx in constituent_indices)
            if has_soup and has_content:
                implicit_detected = True
        elif dish_key in {'hanamakibaozi', 'wontondumplings', 'pie'}:
            has_crust = any(foods[idx]['source_key'] == 'bread' for idx in constituent_indices)
            has_filling = len(constituent_indices) >= 2
            if has_crust and has_filling:
                implicit_detected = True

        if implicit_detected:
            for idx in constituent_indices:
                consumed_indices.add(idx)

            dish_name = display_name(dish_key)
            summary = f"Inferred {dish_name} from detected ingredients and combined them as one meal."
            group_foods = [foods[idx] for idx in constituent_indices]
            
            combined = sum_foods(dish_name, group_foods, summary)
            combined['source_key'] = dish_key
            processed_dishes.append(combined)
            combo_notes.append(summary)

    remaining_foods = [item for i, item in enumerate(foods) if i not in consumed_indices]
    all_current_foods = processed_dishes + remaining_foods
    by_key = {item['source_key']: item for item in all_current_foods}

    final_foods = []
    used_keys = set()

    for rule in COMBO_RULES:
        if rule['keys'].issubset(by_key):
            combo_items = [by_key[key] for key in rule['keys']]
            final_foods.append(sum_foods(rule['name'], combo_items, rule['summary']))
            used_keys.update(rule['keys'])
            combo_notes.append(rule['summary'])

    for item in all_current_foods:
        if item['source_key'] not in used_keys:
            final_foods.append(item)

    return final_foods, ' '.join(combo_notes) if combo_notes else None


def calculate_totals(foods):
    totals = {'calories': 0, 'protein_g': 0, 'carbs_g': 0, 'fat_g': 0, 'fiber_g': 0}
    for item in foods:
        totals['calories'] += item['calories']
        totals['protein_g'] += item['protein_g']
        totals['carbs_g'] += item['carbs_g']
        totals['fat_g'] += item['fat_g']
        totals['fiber_g'] += item['fiber_g']
    return {key: safe_number(value) for key, value in totals.items()}


def resolve_model_path():
    if len(sys.argv) >= 3:
        return Path(sys.argv[2])

    env_model_path = os.getenv('FOOD_MODEL_PATH')
    if env_model_path:
        return Path(env_model_path)

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    return FALLBACK_MODEL_PATH


FOOD_RECOMMENDATIONS = {
    'pizza': [
        'Pizza has high saturated fat; balance it with a side salad.',
        'Watch portion sizes of pizza to manage overall sodium intake.'
    ],
    'hamburg': [
        'Burgers can be high in calories. Choose a lean protein patty if possible.',
        'Avoid extra cheese or high-fat sauces to keep the burger lighter.'
    ],
    'dosa': [
        'Dosa is a fermented crepe, which is excellent for gut health.',
        'Try to limit high-calorie coconut chutney and pair it with sambar.'
    ],
    'biryani': [
        'Biryani is rich and calorie-dense; try pairing it with cucumber raita.',
        'Balance biryani with a protein-rich side salad.'
    ],
    'dal': [
        'Dal is a great source of plant protein and fiber.',
        'Add a squeeze of lemon to your dal to enhance iron absorption.'
    ],
    'roti': [
        'Roti provides healthy complex carbs. Avoid adding excess butter/ghee.',
        'Pair roti with high-protein lentils or paneer for a balanced meal.'
    ],
    'rice': [
        'Rice is a great source of fast energy. Control the portion size.',
        'Pair white rice with fiber-rich vegetables to lower its glycemic index.'
    ],
    'samosa': [
        'Samosas are deep-fried; limit to one and enjoy as an occasional treat.',
        'Pair samosa with green mint chutney rather than sweet tamarind sauce.'
    ],
    'idli': [
        'Idli is steamed and fat-free, making it a very light, healthy meal.',
        'Enjoy idli with vegetable-rich sambar for added fiber and protein.'
    ]
}


def build_recommendations(total, foods):
    if not foods:
        return [
            'We couldn\'t identify any food items clearly. Please try again with a clearer picture.',
            'Ensure the food is well-lit and not covered or partially hidden.',
            'Taking the photo from directly above (top-down angle) helps the model detect items.'
        ]

    recommendations = []
    
    # 1. Add food-specific recommendations first
    for item in foods:
        skey = item.get('source_key', '').lower()
        if skey in FOOD_RECOMMENDATIONS:
            for rec in FOOD_RECOMMENDATIONS[skey]:
                if rec not in recommendations:
                    recommendations.append(rec)
                    break
    
    # 2. Add macro-based recommendations dynamically if we don't have enough food-specific ones
    if len(recommendations) < 3:
        if total.get('calories', 0) > 700:
            rec = 'Watch portion sizes to keep the meal balanced.'
            if rec not in recommendations:
                recommendations.append(rec)
        if total.get('fiber_g', 0) < 8:
            rec = 'Add vegetables, salad, or whole grains to increase fiber.'
            if rec not in recommendations:
                recommendations.append(rec)
        if total.get('protein_g', 0) < 15:
            rec = 'Add dal, paneer, curd, eggs, or another protein source.'
            if rec not in recommendations:
                recommendations.append(rec)

    # 3. Fallbacks if list is still short
    fallback_recs = [
        'Drink plenty of water before and during the meal for better digestion.',
        'Eat slowly and mindfully to help recognize when you are full.',
        'Pair this meal with physical activity later in the day to maintain balance.'
    ]
    for rec in fallback_recs:
        if len(recommendations) >= 3:
            break
        if rec not in recommendations:
            recommendations.append(rec)

    return recommendations[:3]


def compute_health_score(foods, total):
    if not foods:
        return 4

    healthy_count = sum(1 for item in foods if item.get('healthy'))
    score = 6 + healthy_count
    if total['calories'] > 900:
        score -= 2
    elif total['calories'] > 700:
        score -= 1
    if total['fiber_g'] >= 10:
        score += 1
    return max(1, min(10, score))


def estimate_food(food_name, box_area, image_area, confidence, nutrition_table):
    source_key = canonical_key(food_name)
    nutrition = nutrition_table.get(canonical_key(food_name), fallback_nutrition(food_name))
    
    # Calculate box_ratio (percentage of image area occupied by the box)
    box_ratio = box_area / image_area
    # Map box_ratio (typically 0.01 to 0.40) to a conservative size_factor range (0.5 to 1.8)
    # A standard single item usually occupies between 10% and 30% of the image.
    if box_ratio > 0.15:
        size_factor = 1.0 + (box_ratio - 0.15) * 2.0
    else:
        size_factor = 0.6 + (box_ratio / 0.15) * 0.4
    
    # Cap size_factor between 0.5 and 1.8
    size_factor = min(1.8, max(0.5, size_factor))

    return {
        'name': display_name(food_name),
        'estimated_weight_g': safe_number(nutrition['weight_g'] * size_factor),
        'calories': safe_number(nutrition['calories'] * size_factor),
        'protein_g': safe_number(nutrition['protein_g'] * size_factor),
        'carbs_g': safe_number(nutrition['carbs_g'] * size_factor),
        'fat_g': safe_number(nutrition['fat_g'] * size_factor),
        'fiber_g': safe_number(nutrition['fiber_g'] * size_factor),
        'confidence': safe_number(confidence),
        'healthy': bool(nutrition.get('healthy')),
        'source_key': source_key
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Missing image path argument.'}))
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = resolve_model_path()
    if not model_path.exists():
        print(json.dumps({'error': f'Model file not found: {model_path}'}))
        sys.exit(1)

    nutrition_table = load_nutrition()
    model = YOLO(str(model_path))
    results = model(image_path, imgsz=640, conf=0.15, device='cpu', verbose=False)[0]

    height, width = results.orig_shape
    image_area = width * height if width and height else 1
    foods = []
    seen = set()

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            name = results.names[int(cls)].lower().strip()
            if canonical_key(name) in IGNORED_CLASSES:
                continue
            if name in seen:
                continue

            x1, y1, x2, y2 = box
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            foods.append(estimate_food(name, box_area, image_area, float(conf), nutrition_table))
            seen.add(name)

    if not foods:
        print(json.dumps({
            'meal_name': 'No Food Detected',
            'foods': [],
            'total': {'calories': 0, 'protein_g': 0, 'carbs_g': 0, 'fat_g': 0, 'fiber_g': 0},
            'health_score': 4,
            'summary': 'Local YOLO did not clearly detect food classes in the image. Try uploading a clearer, top-down view of the meal.',
            'recommendations': build_recommendations({}, [])
        }))
        return

    foods, combo_note = postprocess_foods(foods)
    totals = calculate_totals(foods)
    meal_name = ' + '.join(item['name'] for item in foods[:3])
    base_summary = f'Local YOLO model {model_path.name} detected meal-level food classes and estimated nutrition from the local table.'
    if combo_note:
        base_summary = f'{combo_note} {base_summary}'

    print(json.dumps({
        'meal_name': meal_name,
        'foods': [
            {
                key: value
                for key, value in item.items()
                if key not in {'healthy', 'source_key', 'source_keys', 'composition_note'}
            }
            for item in foods
        ],
        'total': totals,
        'health_score': compute_health_score(foods, totals),
        'summary': base_summary,
        'recommendations': build_recommendations(totals, foods)
    }))


if __name__ == '__main__':
    main()
