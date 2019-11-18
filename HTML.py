import tornado.ioloop
import tornado.web
import os
import uuid


class NewHandler(tornado.web.RequestHandler):

    def get(self):
        #n = int(self.get_argument("n"))  # 获取url的参数值
        #self.write(str(self.service.calc(n)))  # 使用阶乘服务
        self.render("in.html", title="My title", items=["Calorie", "Predictor"])
"""
class KHandler(tornado.web.RequestHandler):
    def get(self):
        #n = int(self.get_argument("n"))  # 获取url的参数值
        #self.write(str(self.service.calc(n)))  # 使用阶乘服务
        self.render("index.html", title="My title", items=["c","p"])
"""

def uuid_naming_strategy(original_name):
    "File naming strategy that ignores original name and returns an UUID"
    return str(uuid.uuid4())

class UploadFileHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", title="My title", items=["Calorie", "Predictor"])

    def post(self):
        #文件的暂存路径
        upload_path=os.path.join(os.path.dirname(__file__),'file.jpg')
        #提取表单中‘name’为‘file’的文件元数据
        file_metas=self.request.files['file']
        for meta in file_metas:
            filename=meta['filename']
            filepath=os.path.join(upload_path,filename)
            #有些文件需要已二进制的形式存储，实际中可以更改
            with open(upload_path,'wb') as up:
                up.write(meta['body'])

        self.render("index.html", title="My title", items=["c", "p"])

class DownloadHandler(tornado.web.RequestHandler):
    def post(self, filename):
        print('i download file handler : ', filename)
        # Content-Type这里我写的时候是固定的了，也可以根据实际情况传值进来
        self.set_header('Content-Type', 'application/octet-stream')
        self.set_header('Content-Disposition', 'attachment; filename=' + filename)
        # 读取的模式需要根据实际情况进行修改
        with open(filename, 'rb') as f:
            while True:
                data = f.read(4096)
                if not data:
                    break
                self.write(data)
        # 记得要finish
        self.finish()
    get = post

def make_app():
    return tornado.web.Application([
        (r"/", NewHandler),
        (r"/k",UploadFileHandler)],# 注册路由
        static_path=os.path.join(os.path.dirname(__file__), "static"),
    )
if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
