import h5py
import numpy as np
from PuzzleLib import Config
from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.Kernels.Embedder import embed, embedBackwardParams
from PuzzleLib.Backend.Kernels import MatVec
from PuzzleLib.Variable import Variable
from PuzzleLib.Modules.Module import ModuleError, Module
from PuzzleLib.Modules import Reshape


class Embedder(Module):
    def __init__(self, vocabulary, embsize, onVocabulary=None, initscheme="uniform", wscale=1.0,
                 learnable=True, name=None):
        super().__init__(name)
        args = dict(locals())

        self.embsize = embsize

        self.wgrad = None
        self.learnable = learnable
        self.outgrad = None

        dt = h5py.special_dtype(vlen=str)

        if isinstance(vocabulary, dict):
            vocabsize = len(vocabulary)
            vocab = np.empty(shape=(vocabsize, ), dtype=dt)

            for word, idx in vocabulary.items():
                vocab[int(idx)] = word

        elif isinstance(vocabulary, int):
            vocabsize = vocabulary
            vocab = np.empty(shape=(0, ), dtype=dt)

        else:
            raise ModuleError("Unrecognized vocabulary parameter type")

        self.vocab = None
        self.setAttr("vocab", vocab)

        args["vocabulary"] = vocabsize
        self.registerBlueprint(args, exclude=["onVocabulary"])

        Wshape = (vocabsize, embsize)
        W = self.createTensorWithScheme(initscheme, Wshape, wscale, (embsize, vocabsize))
        if W is None:
            W = np.empty(Wshape, dtype=np.float32)

        if onVocabulary is not None:
            onVocabulary(W)

        self.W = None
        self.setVar("W", Variable(gpuarray.to_gpu(W)))

        self.loadVarHook = self.checkVarOnLoad
        self.loadAttrHook = self.checkAttrOnLoad

    def checkVarOnLoad(self, paramName, dataset):
        if paramName == "W":
            if dataset.shape[1] != self.embsize:
                raise ModuleError("Expected embedding size %s, was given %s" % (self.embsize, dataset.shape[1]))

            self.setVar("W", Variable(gpuarray.to_gpu(dataset)))

        else:
            raise ModuleError("Unknown parameter name '%s' for embedder" % paramName)

    def checkAttrOnLoad(self, attrName, dataset):
        if attrName == "vocab":
            self.setAttr("vocab", dataset)

        else:
            raise ModuleError("Unknown attribute name '%s' for embedder" % attrName)

    def getVocabulary(self):
        voc = {}

        if self.hasAttr("vocab"):
            for i in range(self.vocab.shape[0]):
                voc[self.vocab[i]] = i

        return voc

    def verifyData(self, data):
        mn, mx = gpuarray.minimum(data).get(), gpuarray.maximum(data).get()
        if mn < -1:
            raise ModuleError("Embedder data verification failed, found index %s (< -1)" % mn)

        if mx >= self.W.shape[0]:
            raise ModuleError("Embedder data verification failed, found index %s (vocabulary size is %s)" %
                              (mx, self.W.shape[0]))

    def updateData(self, data):
        if Config.verifyData:
            self.verifyData(data)
        self.data = embed(data, self.W)

    def updateGrad(self, grad):
        self.grad = None

    def accGradParams(self, grad, scale=1.0, momentum=0.0):
        self.outgrad = grad
        self.vars["W"].grad.fill(0.0)

        if self.learnable:
            embedBackwardParams(self.inData, grad, self.vars["W"].grad, scale)

    def updateParams(self, learnRate):
        if self.learnable:
            embedBackwardParams(self.inData, self.outgrad, self.W, learnRate)

    def dataShapeFrom(self, shape):
        batchsize, sentlen = shape
        return batchsize, sentlen, self.embsize

    def gradShapeFrom(self, shape):
        raise ModuleError("Gradient propagation is undefined")

    def checkDataShape(self, shape):
        if len(shape) != 2:
            raise ModuleError("Data must be 2d matrix")

    def checkGradShape(self, shape):
        if len(shape) != 3:
            raise ModuleError("Grad must be 3d tensor")

        batchsize, sentlen, embsize = shape

        if embsize != self.embsize:
            raise ModuleError("Expected %d grad embedding size, %d was given" % (self.embsize, embsize))

        if batchsize != self.inData.shape[0]:
            raise ModuleError("Expected %d grad batch size, %d was given" % (self.inData.shape[0], batchsize))

    def checkDataType(self, dtype):
        if dtype != np.int32:
            raise ModuleError("Expected int32-tensor (got dtype %s)" % dtype)

    def reset(self):
        super().reset()
        self.outgrad = None
        
    def calcMode(self, T):
        if self.calctype == T:
            return

        variables = self.vars
        self.vars = {}

        for varName, var in variables.items():
            self.setVar(varName, Variable(var.data.astype(T), name=var.name, grad=var.grad.astype(T)))

        self.calctype = T
        

class Linear(Module):
    def __init__(self, insize, outsize, wscale=1.0, useBias=True, initscheme=None, name=None,
                 empty=False, transpose=False):
        super().__init__(name)
        self.registerBlueprint(locals())

        self.transpose = transpose
        self.useBias = useBias

        self.W = None
        self.b = None

        if empty:
            return

        Wshape, bshape = ((outsize, insize), (insize, )) if transpose else ((insize, outsize), (outsize, ))
        W = self.createTensorWithScheme(initscheme, Wshape, wscale, factorShape=Wshape)

        self.setVar("W", Variable(gpuarray.empty(Wshape, dtype=self.calctype) if W is None else gpuarray.to_gpu(W)))

        if useBias:
            self.setVar("b", Variable(gpuarray.zeros(bshape, dtype=self.calctype)))

    def updateData(self, data):
        reshape = len(data.shape)>2
        if reshape:
            reshape2d = Reshape((int(np.prod(data.shape[:-1])), data.shape[-1]))
            reshape2d.calcMode(self.calctype)
            reshapeNd = Reshape(data.shape[:-1] + tuple([self.W.shape[1]]))
            reshapeNd.calcMode(self.calctype)
            data = reshape2d(data)
        self.data = Blas.mulMatrixOnMatrix(data, self.W, transpB=self.transpose)
        if self.useBias:
            MatVec.addVecToMat(self.b, self.data, axis=1, out=self.data)
        if reshape:
            self.data = reshapeNd(self.data)

    def updateGrad(self, grad):
        self.grad = Blas.mulMatrixOnMatrix(grad, self.W, transpB=not self.transpose)

    def accGradParams(self, grad, scale=1.0, momentum=0.0):
        if not self.transpose:
            Blas.mulMatrixOnMatrix(self.inData, grad, out=self.vars["W"].grad, transpA=True, alpha=scale, beta=momentum)
        else:
            Blas.mulMatrixOnMatrix(grad, self.inData, out=self.vars["W"].grad, transpA=True, alpha=scale, beta=momentum)

        if self.useBias:
            Blas.sumOnMatrix(grad, out=self.vars["b"].grad, alpha=scale, beta=momentum)

    def dataShapeFrom(self, shape):
        return (shape[0], self.W.shape[1]) if not self.transpose else (shape[0], self.W.shape[0])

    def checkDataShape(self, shape):
        if not self.transpose:
            if shape[-1] != self.W.shape[0]:
                raise ModuleError("Expected %d data dimensions, %d were given" % (self.W.shape[0], shape[1]))
        else:
            if shape[-1]!= self.W.shape[1]:
                raise ModuleError("Expected %d data dimensions, %d were given" % (self.W.shape[1], shape[1]))

    def gradShapeFrom(self, shape):
        return (shape[0], self.W.shape[0]) if not self.transpose else (shape[0], self.W.shape[1])

    def checkGradShape(self, shape):
        if len(shape) != 2:
            raise ModuleError("Grad must be 2d matrix")

        if not self.transpose:
            if shape[1] != self.W.shape[1]:
                raise ModuleError("Expected %d grad dimensions, %d were given" % (self.W.shape[1], shape[1]))
        else:
            if shape[1] != self.W.shape[0]:
                raise ModuleError("Expected %d grad dimensions, %d were given" % (self.W.shape[0], shape[1]))

    def calcMode(self, T):
        if self.calctype == T:
            return

        variables = self.vars
        self.vars = {}

        for varName, var in variables.items():
            self.setVar(varName, Variable(var.data.astype(T), name=var.name, grad=var.grad.astype(T)))

        self.calctype = T

