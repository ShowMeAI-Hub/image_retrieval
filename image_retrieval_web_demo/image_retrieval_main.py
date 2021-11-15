import os
import cv2
import time
from datetime import timedelta
from resize_images import resize_images
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify, flash
from retrieval import load_model, load_data, extract_feature, load_query_image, sort_img, extract_feature_query

# 构建调整大小后的图片
create_thumb_images(full_folder='./static/image_database/',
                    thumb_folder='./static/resized_images/',
                    suffix='',
                    height=200,
                    del_former_thumb=True,
                    )

# 准备数据集
data_loader = load_data(data_path='./static/image_database/',
                        batch_size=2,
                        shuffle=False,
                        transform='default',
                        )

# 加载预训练模型
model = load_model(pretrained_model='./retrieval/models/pretrained_model.pth', use_gpu=True)

# 抽取图像特征
gallery_feature, image_paths = extract_feature(model=model, dataloaders=data_loader)

# 支持的图片后缀名
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# Set static file cache expiration time
# app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/', methods=['POST', 'GET'])  # add route
def image_retrieval():

    basepath = os.path.dirname(__file__)    # 当前路径
    upload_path = os.path.join(basepath, 'static/upload_image','query.jpg')

    if request.method == 'POST':
        if request.form['submit'] == 'upload':
            if len(request.files) == 0:
                return render_template('upload_finish.html', message='Please select a picture file!')
            else:
                f = request.files['picture']
         
                if not (f and allowed_file(f.filename)):
                    # return jsonify({"error": 1001, "msg": "Examine picture extension, only png, PNG, jpg, JPG, or bmp supported."})
                    return render_template('upload_finish.html', message='Examine picture extension, png、PNG、jpg、JPG、bmp support.')
                else:

                    f.save(upload_path)
             
                    # transform image format and name with opencv.
                    img = cv2.imread(upload_path)
                    cv2.imwrite(os.path.join(basepath, 'static/upload_image', 'query.jpg'), img)
             
                    return render_template('upload_finish.html', message='Upload successfully!')

        elif request.form['submit'] == 'retrieval':
            start_time = time.time()
            # 加载检索图片
            query_image = load_query_image('./static/upload_image/query.jpg')
            # 抽取特征
            query_feature = extract_feature_query(model=model, img=query_image)
            # 排序
            similarity, index = sort_img(query_feature, gallery_feature)
            sorted_paths = [image_paths[i] for i in index]

            print(sorted_paths)
            small_images = ['./static/resized_images/' + os.path.split(sorted_path)[1] for sorted_path in sorted_paths]
            # sorted_files = [os.path.split(sorted_path)[1] for sorted_path in sorted_paths]

            return render_template('retrieval.html', message="检索完毕, 耗时 {:3f} 秒".format(time.time() - start_time),
            	sml1=similarity[0], sml2=similarity[1], sml3=similarity[2], sml4=similarity[3], sml5=similarity[4], sml6=similarity[5],
            	img1_tmb=small_images[0], img2_tmb=small_images[1],img3_tmb=small_images[2],img4_tmb=small_images[3],img5_tmb=small_images[4],img6_tmb=small_images[5])

    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=8080, debug=True)