import copy
import json
import math
import numpy as np
from bert_punctuator.modules import Embedder, Linear
from PuzzleLib.Backend.Blas import mulTensorBatch
from PuzzleLib.Backend import gpuarray
from PuzzleLib.Modules import Module, Activation, SwapAxes, Mul, SoftMax, ModuleError, Tile, Gelu, BatchNorm, InstanceNorm2D
from PuzzleLib.Backend.Kernels import MatVec
from PuzzleLib.Variable import Variable
from PuzzleLib.Containers import Container, Sequential


class BertConfig(object):
    def __init__(self,
                 vocab_size_or_json,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 segment_size=32):

        if isinstance(vocab_size_or_json, str):
            with open(vocab_size_or_json, "r") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_json, int):
            self.vocab_size = vocab_size_or_json
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.segment_size = segment_size
            self.output_size = output_size
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    def __repr__(self):
        return str(self.to_json())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    
class BertLayerNorm(Module):
    def __init__(self, config, epsilon=1e-12, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        
        self.epsilon = epsilon
        self.scale = None
        self.setVar("scale", Variable(gpuarray.to_gpu(np.ones(config.hidden_size, dtype=np.float32))))
        self.bias = None
        self.setVar("bias", Variable(gpuarray.to_gpu(np.zeros(config.hidden_size, dtype=np.float32))))
        self.mul = Mul()
        
    def updateData(self, data):
        batchsize, maps, h = data.shape
        
        data = data.reshape((batchsize, maps, h, 1))
        norm = InstanceNorm2D(maps, epsilon=self.epsilon)
        norm.calcMode(self.calctype)
        data = norm(data)
        data = data.reshape((batchsize * maps, h))

        tile = Tile(axis=0, times=data.shape[0])
        tile.calcMode(self.calctype)
        scale = tile(self.scale.reshape(tuple([1]) + self.scale.shape))
        
        data = self.mul([data, scale])
        data = MatVec.addVecToMat(self.bias, data, axis=1, out=data)
        self.data = data.reshape(batchsize, maps, h)
        
    def checkDataType(self, dtype):
        if dtype != self.calctype:
            raise ModuleError("Expected {} (got dtype {})".format(self.calctype, dtype))
        
    def calcMode(self, T):
        if self.calctype == T:
            return

        self.mul.calcMode(T)

        variables = self.vars
        self.vars = {}

        for varName, var in variables.items():
            self.setVar(varName, Variable(var.data.astype(T), name=var.name, grad=var.grad.astype(T)))

        attrs = self.attrs
        self.attrs = {}

        for attrName, attr in attrs.items():             
            self.setAttr(attrName, attr.astype(T))               

        self.mul.calctype = T
        self.calctype = T


class BertEmbeddings(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        
        self.append(Embedder(config.vocab_size, config.hidden_size, name='wordEmbedder'))
        self.append(Embedder(config.max_position_embeddings, config.hidden_size, name='positionEmbedder'))
        self.append(Embedder(config.type_vocab_size, config.hidden_size, name='tokenTypeEmbedder'))

        self.append(BertLayerNorm(config, name='LayerNorm'))

    def updateData(self, data):
        if self.acquireDtypesFrom(data) == np.int32:
            inputIds = data
            tokenTypeIds = gpuarray.zeros(inputIds.shape, dtype=np.int32)
        else:
            inputIds, tokenTypeIds = data

        seqlength = inputIds.shape[1]
        positionIds = gpuarray.to_gpu(np.array([range(seqlength)]*inputIds.shape[0]).astype(np.int32))

        wordsEmbeddings = self.modules['wordEmbedder'](inputIds)
        positionEmbeddings = self.modules['positionEmbedder'](positionIds)
        tokenTypeEmbeddings = self.modules['tokenTypeEmbedder'](tokenTypeIds)

        embeddings = wordsEmbeddings + positionEmbeddings + tokenTypeEmbeddings
        embeddings = self.modules['LayerNorm'](embeddings)
        self.data = embeddings
    
    def checkDataType(self, dtype):
        if dtype != np.int32 and dtype != [np.int32, np.int32]:
            raise ModuleError("Expected int32-tensor or [int32-tensor, int32-tensor] (got dtype %s)" % dtype)


class BertSelfAttention(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
            
        self.num_attention_heads = config.num_attention_heads
        self.attentionHeadSize = int(config.hidden_size / config.num_attention_heads)
        self.allHeadSize = self.num_attention_heads * self.attentionHeadSize

        self.append(Linear(config.hidden_size, self.allHeadSize, name='query'))
        self.append(Linear(config.hidden_size, self.allHeadSize, name='key'))
        self.append(Linear(config.hidden_size, self.allHeadSize, name='value'))
        self.append(Mul(name='mul'))
        
    def transpose(self, x):
        x = x.reshape(x.shape[:-1] + (self.num_attention_heads, self.attentionHeadSize))
        swap = SwapAxes(axis1=1, axis2=2)
        swap.calcMode(self.calctype)
        x = swap(x)
        return x

    def updateData(self, data):
        hiddenStates, attentionMask = data
        
        mixedQueryLayer = self.modules['query'](hiddenStates)
        mixedKeyLayer = self.modules['key'](hiddenStates)
        mixedValueLayer = self.modules['value'](hiddenStates)
        
        queryLayer = self.transpose(mixedQueryLayer)
        keyLayer = self.transpose(mixedKeyLayer)
        valueLayer = self.transpose(mixedValueLayer)
        
        batchsize, maps, h, w = queryLayer.shape
        
        swap = SwapAxes(axis1=2, axis2=1)
        swap.calcMode(self.calctype)
        
        A = queryLayer.reshape((batchsize * maps, h, w))
        B = swap(keyLayer.reshape((batchsize * maps, h, w)))
        attentionScores = mulTensorBatch(A, B, formatA="gbp", formatB="gbp", formatOut="gbp")
        attentionScores = attentionScores.reshape((batchsize, maps, h, h))
        
        a = gpuarray.empty(attentionScores.shape, self.calctype).fill(1/math.sqrt(self.attentionHeadSize))
        attentionScores = self.modules['mul']([attentionScores, a])
        attentionScores = attentionScores + attentionMask
        
        softmax = SoftMax()
        softmax.calcMode(self.calctype)
        swap2 = SwapAxes(axis1=1, axis2=3)
        swap2.calcMode(self.calctype)
        attentionProbs = swap2(softmax(swap2(attentionScores)))

        contextLayer = mulTensorBatch(attentionProbs.reshape((batchsize * maps, h, h)), \
                                      valueLayer.reshape((batchsize * maps, h, w)), \
                                      formatA="gbp", formatB="gbp", formatOut="gbp")

        contextLayer = swap(contextLayer.reshape((batchsize, maps, h, w))).reshape((batchsize, h, self.allHeadSize))
        self.data = contextLayer
        
    def calcMode(self, T):
        for mod in self.modules.values():
            try:
                mod.calcMode(T)

            except Exception as e:
                self.handleError(mod, e)
        self.calctype = T

        
class BertSelfOutput(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        
        self.append(Linear(config.hidden_size, config.hidden_size, name='dense'))
        self.append(BertLayerNorm(config, name='LayerNorm'))

    def updateData(self, data):
        hiddenStates, inputTensor = data
        hiddenStates = self.modules['dense'](hiddenStates)
        hiddenStates = self.modules['LayerNorm'](hiddenStates + inputTensor)
        self.data = hiddenStates


class BertAttention(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        
        self.append(BertSelfAttention(config, name='self'))
        self.append(BertSelfOutput(config, name='output'))

    def updateData(self, data):
        inputTensor, attentionMask = data
        selfOutput = self.modules['self']((inputTensor, attentionMask))
        attentionOutput = self.modules['output']((selfOutput, inputTensor))
        self.data = attentionOutput


class BertIntermediate(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        
        self.append(Linear(config.hidden_size, config.intermediate_size, name='dense'))
        self.append(Gelu(name='gelu'))

    def updateData(self, hiddenStates):
        hiddenStates = self.modules['dense'](hiddenStates)
        hiddenStates = self.modules['gelu'](hiddenStates)
        self.data = hiddenStates

    
class BertOutput(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        
        self.append(Linear(config.intermediate_size, config.hidden_size, name='dense'))
        self.append(BertLayerNorm(config, name='LayerNorm'))

    def updateData(self, data):
        hiddenStates, inputTensor = data
        hiddenStates = self.modules['dense'](hiddenStates)
        hiddenStates = self.modules['LayerNorm'](hiddenStates + inputTensor)
        self.data = hiddenStates


class BertLayer(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        
        self.append(BertAttention(config, name='attention'))
        self.append(BertIntermediate(config, name='intermediate'))
        self.append(BertOutput(config, name='output'))

    def updateData(self, data):
        hiddenStates, attentionMask = data
        attentionOutput = self.modules['attention']((hiddenStates, attentionMask))
        intermediateOutput = self.modules['intermediate'](attentionOutput)
        layerOutput = self.modules['output']((intermediateOutput, attentionOutput))
        self.data = layerOutput


class BertEncoder(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        for i in range(config.num_hidden_layers):
            self.append(BertLayer(config, name=i))    

    def updateData(self, data):
        hiddenStates, attentionMask = data
        for i in self.modules:
            hiddenStates = self.modules[i]((hiddenStates, attentionMask))
                   
        self.data = hiddenStates
    
    
class BertPooler(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        self.append(Linear(config.hidden_size, config.hidden_size, name='dense'))
        self.activation = Activation('tanh')

    def updateData(self, data):
        firstTokenTensor = data[:, 0].copy()
        pooledOutput = self.modules['dense'](firstTokenTensor)
        pooledOutput = self.activation(pooledOutput)
        self.data = pooledOutput
    
    
class BertModel(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        self.append(BertEmbeddings(config, name='embeddings'))
        self.append(BertEncoder(config, name='encoder'))
        self.append(BertPooler(config, name='pooler'))
        self.num_attention_heads = config.num_attention_heads
        
    def updateData(self, data):
        inputIds = data
        
        attentionMask = np.ones(inputIds.shape)

        attentionMask = (1.0 - attentionMask) * -10000.0
        extendedAttentionMask = np.repeat(np.expand_dims(attentionMask, axis=1), inputIds.shape[1], axis=1)
        extendedAttentionMask = np.repeat(np.expand_dims(extendedAttentionMask, axis=1), self.num_attention_heads, axis=1)
        calctype = self.modules['embeddings'].modules['LayerNorm'].calctype
        extendedAttentionMask = gpuarray.to_gpu(extendedAttentionMask.astype(calctype))
        
        embeddingOutput = self.modules['embeddings'](inputIds)
        sequenceOutput = self.modules['encoder']((embeddingOutput, extendedAttentionMask))

        self.data = sequenceOutput
            
    def checkDataType(self, dtype):
        if dtype != np.int32:
            raise ModuleError("Expected int32-tensor (got dtype %s)" % dtype)

            
class BertLMPredictionHead(Container):
    def __init__(self, config, bertModelEmbeddingWeights, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        
        self.append(Linear(config.hidden_size, config.hidden_size, name='dense'))
        self.append(Gelu(name='gelu'))
        self.append(BertLayerNorm(config, name='LayerNorm'))
        self.append(Linear(bertModelEmbeddingWeights.shape[1], bertModelEmbeddingWeights.shape[0], name='decoder'))
        self.modules['decoder'].setVar('W', Variable(bertModelEmbeddingWeights))
        self.modules['decoder'].setVar('b', Variable(gpuarray.zeros((bertModelEmbeddingWeights.shape[1],), dtype = np.float32)))

    def updateData(self, hiddenStates):
        hiddenStates = self.modules['dense'](hiddenStates)
        hiddenStates = self.modules['gelu'](hiddenStates)
        hiddenStates = self.modules['LayerNorm'](hiddenStates)
        hiddenStates = self.modules['decoder'](hiddenStates)
        self.data = hiddenStates


class BertForMaskedLM(Container):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        self.append(BertModel(config, name='bert'))
        swap = SwapAxes(axis1=0, axis2=1)
        bertModelEmbeddingWeights = swap(self.modules['bert'].modules['embeddings'].modules['wordEmbedder'].W)
        self.append(BertLMPredictionHead(config, bertModelEmbeddingWeights, name='cls'))

    def updateData(self, inputIds):
        sequenceOutput = self.modules['bert'](inputIds)
        predictionScores = self.modules['cls'](sequenceOutput)
        self.data = predictionScores
        
    def checkDataType(self, dtype):
        if dtype != np.int32:
            raise ModuleError("Expected int32-tensor (got dtype %s)" % dtype)

            
class BertPunc(Container):  
    def __init__(self, config, name=None):
        super().__init__(name)
        self.registerBlueprint(locals())
        self.append(BertForMaskedLM(config, name='lm'))
        self.bert_vocab_size = config.vocab_size
        self.segment_size = config.segment_size
        self.output_size = config.output_size
        self.append(BatchNorm(self.segment_size*self.bert_vocab_size, affine=False, name='bn'))
        self.append(Linear(self.segment_size*self.bert_vocab_size, self.output_size, name='dense'))
        self.modules['bn'].evalMode()

    def updateData(self, data):
        data = self.modules['lm'](data)
        data = data.reshape((data.shape[0], int(np.prod(data.shape[1:]))))
        data = self.modules['bn'](data).astype(np.float16)
        self.data = self.modules['dense'](data)
        
    def checkDataType(self, dtype):
        if dtype != np.int32:
            raise ModuleError("Expected int32-tensor (got dtype %s)" % dtype)
            
    def calcMode(self, T):
        for mod in self.modules.values():
            try:
                mod.calcMode(T)

            except Exception as e:
                self.handleError(mod, e)
        self.calctype = T

