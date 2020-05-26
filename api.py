from flask import Flask,request,jsonify,render_template
import os
import predict
import werkzeug
from flask_restful import reqparse,abort,Api,Resource

app=Flask(__name__)
api=Api(app)

class get_boxes(Resource):
    def post(self):
        parse=reqparse.RequestParser()
        parse.add_argument('image',type=werkzeug.datastructures.FileStorage,location='files')
        args=parse.parse_args()
        img=args['image']
        if img is None:
            return "use properly"
        img.save('temp_data/'+str(img.filename))
        path='temp_data/'+str(img.filename)
        predictions=predict.predict('temp_data/'+str(img.filename))
        os.remove(path)
        return jsonify(predictions)

api.add_resource(get_boxes,'/')

if __name__=='__main__':
    app.run()