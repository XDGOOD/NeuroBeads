import numpy as np
from neurobead_core import SpikingBead

# 1. Простой энкодер (как в демо)
def text_to_spikes(text, dim=128):
    vec = np.zeros(dim, dtype=bool)
    for word in text.lower().split():
        vec[hash(word) % dim] = True
    return vec

# 2. Данные для обучения (Dataset)
# Класс 0: Позитив, Класс 1: Негатив
train_data = [
    ("привет это круто", 0),
    ("мне очень нравится", 0),
    ("прекрасный день", 0),
    ("это ужасно и плохо", 1),
    ("я ненавижу это", 1),
    ("полный провал", 1)
]

def train_bead_model():
    print("🧠 Начинаем обучение бусины NeuroBeads...")
    
    # Создаем бусину и готовим её к обучению на 2 класса
    bead = SpikingBead("TrainerBead", n_neurons=128)
    bead.init_readout(n_classes=2)
    
    # Цикл обучения (достаточно всего 5-10 эпох, так как RLS очень быстрый)
    for epoch in range(10):
        total_error = 0
        for text, label in train_data:
            spikes = text_to_spikes(text)
            
            # Прогоняем через динамику бусины
            processed_spikes = bead.forward(spikes)
            
            # Обучаем ридаут
            error = bead.apply_rls(processed_spikes, target_idx=label)
            total_error += error
            
        if epoch % 2 == 0:
            print(f"Эпоха {epoch}, Средняя ошибка: {total_error/len(train_data):.4f}")

    print("\n✅ Обучение завершено!")
    
    # Проверка (Inference)
    test_text = "это очень плохо"
    test_spikes = text_to_spikes(test_text)
    res_spikes = bead.forward(test_spikes)
    prediction = bead.predict(res_spikes)
    
    label = "ПОЗИТИВ" if np.argmax(prediction) == 0 else "НЕГАТИВ"
    print(f"Тест: '{test_text}' -> Результат: {label} (уверенность: {np.max(prediction):.2f})")

if __name__ == "__main__":
    train_bead_model()