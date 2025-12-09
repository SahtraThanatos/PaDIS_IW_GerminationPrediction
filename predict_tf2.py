import tensorflow as tf
import numpy as np
import cv2

def load_model(path):
    print("Loading model:", path)
    model = tf.saved_model.load(path)
    return model

def load_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_expanded = np.expand_dims(img_rgb, axis=0)
    return img, img_expanded

def run_inference(model, img):
    infer = model.signatures['serving_default']
    output = infer(tf.constant(img, dtype=tf.uint8))
    return output

def draw_boxes(orig, output, score_threshold=0.5):
    h, w, _ = orig.shape

    boxes = output['detection_boxes'][0].numpy()
    scores = output['detection_scores'][0].numpy()
    classes = output['detection_classes'][0].numpy().astype(int)

    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue

        y1, x1, y2, x2 = boxes[i]

        cv2.rectangle(
            orig,
            (int(x1 * w), int(y1 * h)),
            (int(x2 * w), int(y2 * h)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            orig,
            f"{classes[i]}: {scores[i]:.2f}",
            (int(x1 * w), int(y1 * h) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    return orig


if __name__ == "__main__":
    MODEL = "/Users/vladmishchuk/Навчання/Магістр/Проєктування та розробка ІС/IHW/GermPredModels/ZeaMays/exported_graphs/saved_model"
    IMAGE = "/Users/vladmishchuk/Desktop/test_image1.jpg"

    model = load_model(MODEL)
    orig, img = load_image(IMAGE)

    output = run_inference(model, img)

    result = draw_boxes(orig, output)
    cv2.imwrite("result.jpg", result)
    print("Saved result.jpg")
