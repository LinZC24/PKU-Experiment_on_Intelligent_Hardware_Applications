from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import joblib
import numpy as np
import os
import pyaudio
import wave
import cv2
import time

def record_audio(wave_out_path,record_second):
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK)
    wf = wave.open(wave_out_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    for i in range(0, int(RATE / CHUNK * record_second)):
        data = stream.read(CHUNK)
        wf.writeframes(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()

# 生成wavdict，key=wavid，value=wavfile
def gen_wavlist(wavpath):
    wavdict = {}
    labeldict = {}
    for (dirpath, dirnames, filenames) in os.walk(wavpath):
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath = os.sep.join([dirpath, filename])
                fileid = filename.strip('.wav')
                wavdict[fileid] = filepath
                label = fileid.split('_')[1]
                labeldict[fileid] = label
    return wavdict, labeldict

# 特征提取，feat = compute_mfcc(wadict[wavid])
def compute_mfcc(file):
    fs, audio = wavfile.read(file)
    mfcc_feat = mfcc(audio)
    return mfcc_feat

class Model():
    def __init__(self, CATEGORY=None, n_comp=3, n_mix = 3, cov_type='diag', n_iter=1000):
        super(Model, self).__init__()
        self.CATEGORY = CATEGORY
        self.category = len(CATEGORY)
        self.n_comp = n_comp
        self.n_mix = n_mix
        self.cov_type = cov_type
        self.n_iter = n_iter
        # 关键步骤，初始化models，返回特定参数模型的列表
        self.models = []
        for k in range(self.category):
            model = hmm.GMMHMM(n_components=self.n_comp, n_mix = self.n_mix, 
                                covariance_type=self.cov_type, n_iter=self.n_iter)
            self.models.append(model)

    # 模型训练
    def train(self, wavdict=None, labeldict=None):
        print("开始训练")
        for k in range(5):
            subdata = []
            model = self.models[k]
            for x in wavdict:
                if labeldict[x] == self.CATEGORY[k]:
                    mfcc_feat = compute_mfcc(wavdict[x])
                    model.fit(mfcc_feat)

    # 使用特定的测试集合进行测试
    def test(self, filepath):
        result = []
        for k in range(self.category):
            subre = []
            label = []
            model = self.models[k]
            mfcc_feat = compute_mfcc(filepath)
            # 生成每个数据在当前模型下的得分情况
            re = model.score(mfcc_feat)
            subre.append(re)
            result.append(subre)
        # 选取得分最高的种类
        print(result)
        result = np.vstack(result).argmax(axis=0)
        # 返回种类的类别标签
        result = [self.CATEGORY[label] for label in result]
        print('识别得到结果：\n',result)


    def save(self, path="models.pkl"):
        # 利用external joblib保存生成的hmm模型
        joblib.dump(self.models, path)


    def load(self, path="models.pkl"):
        # 导入hmm模型
        self.models = joblib.load(path)


# 准备训练所需数据
if __name__=='__main__':
    CATEGORY = ['1', '2', '3', '4', '5']
    wavdict, labeldict = gen_wavlist('train_sample')
    testpath = 'test_sample'
    # testdict, testlabel = gen_wavlist('test_data')
    # 进行训练
    models = Model(CATEGORY=CATEGORY)
    models.train(wavdict=wavdict, labeldict=labeldict)
    models.save()
    models.load()
    testwavdict, testlabeldict = gen_wavlist(testpath)
    for key in testwavdict.keys():
        models.test(testwavdict[key])
        print('真实结果：\n', testlabeldict[key])
    print('test begin!')
    while True:
        print('序号1：turn on the light， 序号2：turn off the light， 序号3：lighter， 序号4：darker， 序号5：glitter')

        a = input('输入1进行语音检测，输入0停止测试,请输入您的选择：')
        if a == '1':
            print('开始')
            time.sleep(0.5)
            record_audio("test.wav",record_second=2)
            models.test('test.wav')
        else:
            exit(0)