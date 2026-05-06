import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = (224, 224)

# ===== PREPROCESS =====
def preprocess_image(path):
    # Sử dụng tf.keras.utils thay cho thư viện preprocessing cũ
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)

    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    img = preprocess_input(img)
    return img


# ===== BUILD GRAD MODEL =====
def build_grad_model(model, last_conv_layer_name):
    # Tách mô hình để lấy output của layer conv cuối và output dự đoán
    return tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )


# ===== GRAD-CAM++ =====
def gradcam_plus_plus(grad_model, img_array):
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)

        # Xử lý nếu output trả về là dạng list
        if isinstance(preds, list):
            preds = preds[0]

        preds = tf.convert_to_tensor(preds)

        class_idx = tf.argmax(preds[0])
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, conv_output)

    conv_output = conv_output[0]
    grads = grads[0]

    # ===== Tính toán theo công thức Grad-CAM++ =====
    grads_2 = grads ** 2
    grads_3 = grads ** 3

    sum_activations = tf.reduce_sum(conv_output, axis=(0, 1))

    eps = 1e-8
    alpha = grads_2 / (2 * grads_2 + sum_activations * grads_3 + eps)

    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(0, 1))

    heatmap = tf.reduce_sum(weights * conv_output, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap.numpy()

    # Chuẩn hóa heatmap về khoảng [0, 1]
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # Trả về heatmap, index phân lớp và raw_preds để API tính độ tin cậy
    return heatmap, int(class_idx), preds[0].numpy()