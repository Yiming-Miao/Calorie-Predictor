import tornado.ioloop
import tornado.web
import os
import uuid
import json

dictFood={'burger':5.4, 'french_fries':5.08 ,'chicken':3.88, 'toast':3.125, 'egg':1.95, 'pizza':2.66, 'cookie':5.1, 'hot dog':2.9, 'steak':2.7}

class NewHandler(tornado.web.RequestHandler):

    def get(self):
        #n = int(self.get_argument("n"))  
        #self.write(str(self.service.calc(n))) 
        self.render("in.html", title="My title", items=["Calorie", "Predictor"])

class WHandler(tornado.web.RequestHandler):
    def post(self):
        result = 0
        category_path = os.path.join(os.path.dirname(__file__), 'static/categorys.txt')
        file = open(category_path)
        category = eval(file.readlines()[0])
        categorys = []
        for c in category:
            categorys.append(c.strip())
        rst = {}
        weight = {}
        f = open('WeightOutput.txt', 'w')
        for c in categorys:
            i = 0
            temp = self.get_argument(c)
            for j in range(len(temp)):
                if '0' <= temp[j] <= '9':
                    i = i * 10 + int(temp[j]) - int('0')
            rst[c] = i*dictFood[c]
            weight[c] = self.get_argument(c)
            result += rst[c]
            f.writelines(c+":"+str(rst[c])+'\n')


        self.render("hh.html", title="My title", ccc=rst, result=result, weight=weight)

class SHandler(tornado.web.RequestHandler):
    def post(self):
        upload_path = os.path.join(os.path.dirname(__file__), 'static/filess.jpg')
        category_path = os.path.join(os.path.dirname(__file__), 'static/categorys.txt')

        upload_path1 = os.path.join(os.path.dirname(__file__), 'static/file1.jpg')
        file_metas = self.request.files['file']
        for meta in file_metas:
            filename = meta['filename']
            filepath = os.path.join(upload_path1, filename)
            with open(upload_path1, 'wb') as up:
                up.write(meta['body'])


        # self.write(json.dumps(result))
        os.system('cd /Users/mym/Desktop/kaluli/Mask_RCNN-master/samples && python3 demo9.py')
        file = open(category_path)
        category = eval(file.readlines()[0])
        categorys = []
        for c in category:
            categorys.append(c.strip())
        img_path = 'static/filess.jpg'
        result = {
            "categorys": categorys,
            "img_path": img_path
        }
        self.render("second.html", img_path='static/filess.jpg',
                    result=result)


def uuid_naming_strategy(original_name):
    "File naming strategy that ignores original name and returns an UUID"
    return str(uuid.uuid4())

class UploadFileHandler(tornado.web.RequestHandler):
    def get(self):
        upload_path=os.path.join(os.path.dirname(__file__),'static/file1.jpg')
        category_path = os.path.join(os.path.dirname(__file__), 'static/categorys.txt')
        file = open(category_path)

        category = file.readlines()
        categorys = []
        for c in category:
            categorys.append(c.strip())
        img_path = 'static/file1.jpg'
        result = {
            "categorys": categorys,
            "img_path": img_path
        }
        # self.write(json.dumps(result))

        self.render("index.html", title="My title", items=["Calorie", "Predictor"], img_path='static/file1.jpg', result=result)

    def post(self):
        upload_path=os.path.join(os.path.dirname(__file__),'static/file1.jpg')
        file_metas=self.request.files['file']
        for meta in file_metas:
            filename=meta['filename']
            filepath=os.path.join(upload_path,filename)
            with open(upload_path,'wb') as up:
                up.write(meta['body'])
        category_path = os.path.join(os.path.dirname(__file__), 'static/categorys.txt')
        file = open(category_path)

        category = file.readlines()
        categorys= []
        for c in category:
            categorys.append(c.strip())

        img_path = 'static/file1.jpg'
        result = {
            "categorys": categorys,
            "img_path": img_path
        }
        # self.write(json.dumps(result))
        os.system('cd /Users/mym/Desktop/kaluli/Mask_RCNN-master/samples && python3 demo9.py')
        #with open('/Users/mym/Desktop/kaluli/Mask_RCNN-master/samples/demo9.py', 'r') as f:
            #exec(f.read())

        self.render("index.html", title="My title", items=["c", "p"], img_path='static/file1.jpg',result=result)

class DownloadHandler(tornado.web.RequestHandler):
    def post(self, filename):
        print('i download file handler : ', filename)
        self.set_header('Content-Type', 'application/octet-stream')
        self.set_header('Content-Disposition', 'attachment; filename=' + filename)
        with open(filename, 'rb') as f:
            while True:
                data = f.read(4096)
                if not data:
                    break
                self.write(data)
        self.finish()
    get = post


def make_app():
    settings = dict(debug = True)
    return tornado.web.Application([
        (r"/", NewHandler),
        (r"/s", SHandler),
        (r"/k",UploadFileHandler),
         (r"/w", WHandler)],
        **settings,
        # (r"/k",IndexHandler)],
        static_path=os.path.join(os.path.dirname(__file__), "static"),
    )
if __name__ == "__main__":
    app = make_app()
    app.listen(8880)
    tornado.ioloop.IOLoop.current().start()
