import telebot
import config
import numpy as np
import os
import tensorflow as tf
import keras

IMG_SIZE = (160, 160)

# ЗАГРУЗКА МОДЕЛИ
new_model = keras.models.load_model('/home/ubuntu/resolution/MobileNetV2_cat_dog.h5')


bot = telebot.TeleBot(config.token)
#apihelper.proxy = {'https': config.proxy_name}


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        raw= message.photo[2].file_id
    except:
        raw= message.photo[0].file_id
    name = "/home/vgnalex/cat_dog_bot/"+raw+".jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(name,'wb') as new_file:
        new_file.write(downloaded_file)

    #bot.send_message(message.from_user.id, name)

    image = tf.keras.preprocessing.image.load_img(name,grayscale=False, color_mode="rgb", target_size=IMG_SIZE)
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions_ = new_model.predict(input_arr)

    predictions = tf.nn.sigmoid(predictions_)
    predictions = tf.where(predictions < 0.5, 0, 1)

    pr_ = str(predictions.numpy()[0][0])
    if pr_ == 0:
        bot.send_message(message.from_user.id, "Это кошка или кот")
    else:
        bot.send_message(message.from_user.id, "Похоже на собаку")



@bot.message_handler(content_types=['text'])
def echo(message):
    bot.send_message(message.chat.id, "Пришлите фото котика или собачки и я попробую угадать кто это")

if __name__ == '__main__':
    bot.polling(none_stop=True, interval=0)
