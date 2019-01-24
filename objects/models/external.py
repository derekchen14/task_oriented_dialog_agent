'''
Created on Jun 13, 2016

An LSTM decoder - add tanh after cell before output gate

@author: xiul
'''
import math
import numpy as np

def initWeights(n,d):
    """ Initialization Strategy """
    #scale_factor = 0.1
    scale_factor = math.sqrt(float(6)/(n + d))
    return (np.random.rand(n,d)*2-1)*scale_factor

def mergeDicts(d0, d1):
    """ for all k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """
    for k in d1:
        if k in d0: d0[k] += d1[k]
        else: d0[k] = d1[k]


class Decoder:
    def __init__(self, input_size, hidden_size, output_size):
        pass
    
    def get_struct(self):
        return {'model': self.model, 'update': self.update, 'regularize': self.regularize}
    
    
    """ Activation Function: Sigmoid, or tanh, or ReLu"""
    def fwdPass(self, Xs, params, **kwargs):
        pass
    
    def bwdPass(self, dY, cache):
        pass
    
    
    """ Batch Forward & Backward Pass"""
    def batchForward(self, ds, batch, params, predict_mode = False):
        caches = []
        Ys = []
        for i,x in enumerate(batch):
            Y, out_cache = self.fwdPass(x, params, predict_mode = predict_mode)
            caches.append(out_cache)
            Ys.append(Y)
           
        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache
    
    def batchBackward(self, dY, cache):
        caches = cache['caches']
        grads = {}
        for i in xrange(len(caches)):
            single_cache = caches[i]
            local_grads = self.bwdPass(dY[i], single_cache)
            mergeDicts(grads, local_grads) # add up the gradients wrt model parameters
            
        return grads


    """ Cost function, returns cost and gradients for model """
    def costFunc(self, ds, batch, params):
        regc = params['reg_cost'] # regularization cost
        
        # batch forward RNN
        Ys, caches = self.batchForward(ds, batch, params, predict_mode = False)
        
        loss_cost = 0.0
        smooth_cost = 1e-15
        dYs = []
        
        for i,x in enumerate(batch):
            labels = np.array(x['labels'], dtype=int)
            
            # fetch the predicted probabilities
            Y = Ys[i]
            maxes = np.amax(Y, axis=1, keepdims=True)
            e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
            P = e/np.sum(e, axis=1, keepdims=True)
            
            # Cross-Entropy Cross Function
            loss_cost += -np.sum(np.log(smooth_cost + P[range(len(labels)), labels]))
            
            for iy,y in enumerate(labels):
                P[iy,y] -= 1 # softmax derivatives
            dYs.append(P)
            
        # backprop the RNN
        grads = self.batchBackward(dYs, caches)
        
        # add L2 regularization cost and gradients
        reg_cost = 0.0
        if regc > 0:    
            for p in self.regularize:
                mat = self.model[p]
                reg_cost += 0.5*regc*np.sum(mat*mat)
                grads[p] += regc*mat

        # normalize the cost and gradient by the batch size
        batch_size = len(batch)
        reg_cost /= batch_size
        loss_cost /= batch_size
        for k in grads: grads[k] /= batch_size

        out = {}
        out['cost'] = {'reg_cost' : reg_cost, 'loss_cost' : loss_cost, 'total_cost' : loss_cost + reg_cost}
        out['grads'] = grads
        return out


    """ A single batch """
    def singleBatch(self, ds, batch, params):
        learning_rate = params.get('learning_rate', 0.0)
        decay_rate = params.get('decay_rate', 0.999)
        momentum = params.get('momentum', 0)
        grad_clip = params.get('grad_clip', 1)
        smooth_eps = params.get('smooth_eps', 1e-8)
        sdg_type = params.get('sdgtype', 'rmsprop')

        for u in self.update:
            if not u in self.step_cache: 
                self.step_cache[u] = np.zeros(self.model[u].shape)
        
        cg = self.costFunc(ds, batch, params)
        
        cost = cg['cost']
        grads = cg['grads']
        
        # clip gradients if needed
        if params['activation_func'] == 'relu':
            if grad_clip > 0:
                for p in self.update:
                    if p in grads:
                        grads[p] = np.minimum(grads[p], grad_clip)
                        grads[p] = np.maximum(grads[p], -grad_clip)
        
        # perform parameter update
        for p in self.update:
            if p in grads:
                if sdg_type == 'vanilla':
                    if momentum > 0: dx = momentum*self.step_cache[p] - learning_rate*grads[p]
                    else: dx = -learning_rate*grads[p]
                    self.step_cache[p] = dx
                elif sdg_type == 'rmsprop':
                    self.step_cache[p] = self.step_cache[p]*decay_rate + (1.0-decay_rate)*grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                elif sdg_type == 'adgrad':
                    self.step_cache[p] += grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                    
                self.model[p] += dx

        # create output dict and return
        out = {}
        out['cost'] = cost
        return out
    
    
    """ Evaluate on the dataset[split] """
    def eval(self, ds, split, params):
        acc = 0
        total = 0
        
        total_cost = 0.0
        smooth_cost = 1e-15
        perplexity = 0
        
        for i, ele in enumerate(ds.split[split]):
            #ele_reps = self.prepare_input_rep(ds, [ele], params)
            #Ys, cache = self.fwdPass(ele_reps[0], params, predict_model=True)
            #labels = np.array(ele_reps[0]['labels'], dtype=int)
            
            Ys, cache = self.fwdPass(ele, params, predict_model=True)
            
            maxes = np.amax(Ys, axis=1, keepdims=True)
            e = np.exp(Ys - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            
            labels = np.array(ele['labels'], dtype=int)
            
            if np.all(np.isnan(probs)): probs = np.zeros(probs.shape)
            
            log_perplex = 0
            log_perplex += -np.sum(np.log2(smooth_cost + probs[range(len(labels)), labels]))
            log_perplex /= len(labels)
            
            loss_cost = 0
            loss_cost += -np.sum(np.log(smooth_cost + probs[range(len(labels)), labels]))
            
            perplexity += log_perplex #2**log_perplex
            total_cost += loss_cost
            
            pred_words_indices = np.nanargmax(probs, axis=1)
            for index, l in enumerate(labels):
                if pred_words_indices[index] == l:
                    acc += 1
            
            total += len(labels)
            
        perplexity /= len(ds.split[split])    
        total_cost /= len(ds.split[split])
        accuracy = 0 if total == 0 else float(acc)/total
        
        #print ("perplexity: %s, total_cost: %s, accuracy: %s" % (perplexity, total_cost, accuracy))
        result = {'perplexity': perplexity, 'cost': total_cost, 'accuracy': accuracy}
        return result
    
    
         
    """ prediction on dataset[split] """
    def predict(self, ds, split, params):
        inverse_word_dict = {ds.data['word_dict'][k]:k for k in ds.data['word_dict'].keys()}
        for i, ele in enumerate(ds.split[split]):
            pred_ys, pred_words = self.forward(inverse_word_dict, ele, params, predict_model=True)
            
            sentence = ' '.join(pred_words[:-1])
            real_sentence = ' '.join(ele['sentence'].split(' ')[1:-1])
            
            if params['dia_slot_val'] == 2 or params['dia_slot_val'] == 3: 
                sentence = self.post_process(sentence, ele['slotval'], ds.data['slot_dict'])
            
            print('test case', i)
            print('real:', real_sentence)
            print('pred:', sentence)
    
    """ post_process to fill the slot """
    def post_process(self, pred_template, slot_val_dict, slot_dict):
        sentence = pred_template
        suffix = "_PLACEHOLDER"
        
        for slot in slot_val_dict.keys():
            slot_vals = slot_val_dict[slot]
            slot_placeholder = slot + suffix
            if slot == 'result' or slot == 'numberofpeople': continue
            for slot_val in slot_vals:
                tmp_sentence = sentence.replace(slot_placeholder, slot_val, 1)
                sentence = tmp_sentence
                
        if 'numberofpeople' in slot_val_dict.keys():
            slot_vals = slot_val_dict['numberofpeople']
            slot_placeholder = 'numberofpeople' + suffix
            for slot_val in slot_vals:
                tmp_sentence = sentence.replace(slot_placeholder, slot_val, 1)
                sentence = tmp_sentence
                
        for slot in slot_dict.keys():
            slot_placeholder = slot + suffix
            tmp_sentence = sentence.replace(slot_placeholder, '')
            sentence = tmp_sentence
        
        return sentence

class DeepDialogDecoder(Decoder):
    def __init__(self, diaact_input_size, input_size, hidden_size, output_size):
        self.model = {}
        # connections from diaact to hidden layer
        self.model['Wah'] = initWeights(diaact_input_size, 4*hidden_size)
        self.model['bah'] = np.zeros((1, 4*hidden_size))
        
        # Recurrent weights: take x_t, h_{t-1}, and bias unit, and produce the 3 gates and the input to cell signal
        self.model['WLSTM'] = initWeights(input_size + hidden_size + 1, 4*hidden_size)
        # Hidden-Output Connections
        self.model['Wd'] = initWeights(hidden_size, output_size)*0.1
        self.model['bd'] = np.zeros((1, output_size))

        self.update = ['Wah', 'bah', 'WLSTM', 'Wd', 'bd']
        self.regularize = ['Wah', 'WLSTM', 'Wd']

        self.step_cache = {}
        
    """ Activation Function: Sigmoid, or tanh, or ReLu """
    def fwdPass(self, Xs, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        feed_recurrence = params.get('feed_recurrence', 0)
        
        Ds = Xs['diaact']
        Ws = Xs['words']
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = Ds.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        n, xd = Ws.shape
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((n, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((n, d))
        IFOG = np.zeros((n, 4*d))
        IFOGf = np.zeros((n, 4*d)) # after nonlinearity
        Cellin = np.zeros((n, d))
        Cellout = np.zeros((n, d))
    
        for t in xrange(n):
            prev = np.zeros(d) if t==0 else Hout[t-1]
            Hin[t,0] = 1 # bias
            Hin[t, 1:1+xd] = Ws[t]
            Hin[t, 1+xd:] = prev
            
            # compute all gate activations. dots:
            IFOG[t] = Hin[t].dot(WLSTM)
            
            # add diaact vector here
            if feed_recurrence == 0:
                if t == 0: IFOG[t] += Dsh[0]
            else:
                IFOG[t] += Dsh[0]

            IFOGf[t, :3*d] = 1/(1+np.exp(-IFOG[t, :3*d])) # sigmoids; these are three gates
            IFOGf[t, 3*d:] = np.tanh(IFOG[t, 3*d:]) # tanh for input value
            
            Cellin[t] = IFOGf[t, :d] * IFOGf[t, 3*d:]
            if t>0: Cellin[t] += IFOGf[t, d:2*d]*Cellin[t-1]
            
            Cellout[t] = np.tanh(Cellin[t])
            
            Hout[t] = IFOGf[t, 2*d:3*d] * Cellout[t]

        Wd = self.model['Wd']
        bd = self.model['bd']
            
        Y = Hout.dot(Wd)+bd
            
        cache = {}
        if not predict_mode:
            cache['WLSTM'] = WLSTM
            cache['Hout'] = Hout
            cache['WLSTM'] = WLSTM
            cache['Wd'] = Wd
            cache['IFOGf'] = IFOGf
            cache['IFOG'] = IFOG
            cache['Cellin'] = Cellin
            cache['Cellout'] = Cellout
            cache['Ws'] = Ws
            cache['Ds'] = Ds
            cache['Hin'] = Hin
            cache['Dsh'] = Dsh
            cache['Wah'] = Wah
            cache['feed_recurrence'] = feed_recurrence
            
        return Y, cache
    
    """ Forward pass on prediction """
    def forward(self, dict, Xs, params, **kwargs):
        max_len = params.get('max_len', 30)
        feed_recurrence = params.get('feed_recurrence', 0)
        decoder_sampling = params.get('decoder_sampling', 0)
        
        Ds = Xs['diaact']
        Ws = Xs['words']
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = Ds.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        xd = Ws.shape[1]
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((1, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((1, d))
        IFOG = np.zeros((1, 4*d))
        IFOGf = np.zeros((1, 4*d)) # after nonlinearity
        Cellin = np.zeros((1, d))
        Cellout = np.zeros((1, d))
        
        Wd = self.model['Wd']
        bd = self.model['bd']
        
        Hin[0,0] = 1 # bias
        Hin[0,1:1+xd] = Ws[0]
        
        IFOG[0] = Hin[0].dot(WLSTM)
        IFOG[0] += Dsh[0]
        
        IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
        IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
        Cellin[0] = IFOGf[0, :d] * IFOGf[0, 3*d:]
        Cellout[0] = np.tanh(Cellin[0])
        Hout[0] = IFOGf[0, 2*d:3*d] * Cellout[0]
        
        pred_y = []
        pred_words = []
        
        Y = Hout.dot(Wd) + bd
        maxes = np.amax(Y, axis=1, keepdims=True)
        e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
        probs = e/np.sum(e, axis=1, keepdims=True)
            
        if decoder_sampling == 0: # sampling or argmax
            pred_y_index = np.nanargmax(Y)
        else:
            pred_y_index = np.random.choice(Y.shape[1], 1, p=probs[0])[0]
        pred_y.append(pred_y_index)
        pred_words.append(dict[pred_y_index])
        
        time_stamp = 0
        while True:
            if dict[pred_y_index] == 'e_o_s' or time_stamp >= max_len: break
            
            X = np.zeros(xd)
            X[pred_y_index] = 1
            Hin[0,0] = 1 # bias
            Hin[0,1:1+xd] = X
            Hin[0, 1+xd:] = Hout[0]
            
            IFOG[0] = Hin[0].dot(WLSTM)
            if feed_recurrence == 1:
                IFOG[0] += Dsh[0]
        
            IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
            IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
            C = IFOGf[0, :d]*IFOGf[0, 3*d:]
            Cellin[0] = C + IFOGf[0, d:2*d]*Cellin[0]
            Cellout[0] = np.tanh(Cellin[0])
            Hout[0] = IFOGf[0, 2*d:3*d]*Cellout[0]
            
            Y = Hout.dot(Wd) + bd
            maxes = np.amax(Y, axis=1, keepdims=True)
            e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            
            if decoder_sampling == 0:
                pred_y_index = np.nanargmax(Y)
            else:
                pred_y_index = np.random.choice(Y.shape[1], 1, p=probs[0])[0]
            pred_y.append(pred_y_index)
            pred_words.append(dict[pred_y_index])
            
            time_stamp += 1
            
        return pred_y, pred_words
    
    """ Forward pass on prediction with Beam Search """
    def beam_forward(self, dict, Xs, params, **kwargs):
        max_len = params.get('max_len', 30)
        feed_recurrence = params.get('feed_recurrence', 0)
        beam_size = params.get('beam_size', 10)
        decoder_sampling = params.get('decoder_sampling', 0)
        
        Ds = Xs['diaact']
        Ws = Xs['words']
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = Ds.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        xd = Ws.shape[1]
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((1, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((1, d))
        IFOG = np.zeros((1, 4*d))
        IFOGf = np.zeros((1, 4*d)) # after nonlinearity
        Cellin = np.zeros((1, d))
        Cellout = np.zeros((1, d))
        
        Wd = self.model['Wd']
        bd = self.model['bd']
        
        Hin[0,0] = 1 # bias
        Hin[0,1:1+xd] = Ws[0]
        
        IFOG[0] = Hin[0].dot(WLSTM)
        IFOG[0] += Dsh[0]
        
        IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
        IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
        Cellin[0] = IFOGf[0, :d] * IFOGf[0, 3*d:]
        Cellout[0] = np.tanh(Cellin[0])
        Hout[0] = IFOGf[0, 2*d:3*d] * Cellout[0]
        
        # keep a beam here
        beams = [] 
        
        Y = Hout.dot(Wd) + bd
        maxes = np.amax(Y, axis=1, keepdims=True)
        e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
        probs = e/np.sum(e, axis=1, keepdims=True)
        
        # add beam search here
        if decoder_sampling == 0: # no sampling
            beam_candidate_t = (-probs[0]).argsort()[:beam_size]
        else:
            beam_candidate_t = np.random.choice(Y.shape[1], beam_size, p=probs[0])
        #beam_candidate_t = (-probs[0]).argsort()[:beam_size]
        for ele in beam_candidate_t:
            beams.append((np.log(probs[0][ele]), [ele], [dict[ele]], Hout[0], Cellin[0]))
        
        #beams.sort(key=lambda x:x[0], reverse=True)
        #beams.sort(reverse = True)
        
        time_stamp = 0
        while True:
            beam_candidates = []
            for b in beams:
                log_prob = b[0]
                pred_y_index = b[1][-1]
                cell_in = b[4]
                hout_prev = b[3]
                
                if b[2][-1] == "e_o_s": # this beam predicted end token. Keep in the candidates but don't expand it out any more
                    beam_candidates.append(b)
                    continue
        
                X = np.zeros(xd)
                X[pred_y_index] = 1
                Hin[0,0] = 1 # bias
                Hin[0,1:1+xd] = X
                Hin[0, 1+xd:] = hout_prev
                
                IFOG[0] = Hin[0].dot(WLSTM)
                if feed_recurrence == 1: IFOG[0] += Dsh[0]
        
                IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
                IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
                C = IFOGf[0, :d]*IFOGf[0, 3*d:]
                cell_in = C + IFOGf[0, d:2*d]*cell_in
                cell_out = np.tanh(cell_in)
                hout_prev = IFOGf[0, 2*d:3*d]*cell_out
                
                Y = hout_prev.dot(Wd) + bd
                maxes = np.amax(Y, axis=1, keepdims=True)
                e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
                probs = e/np.sum(e, axis=1, keepdims=True)
                
                if decoder_sampling == 0: # no sampling
                    beam_candidate_t = (-probs[0]).argsort()[:beam_size]
                else:
                    beam_candidate_t = np.random.choice(Y.shape[1], beam_size, p=probs[0])
                #beam_candidate_t = (-probs[0]).argsort()[:beam_size]
                for ele in beam_candidate_t:
                    beam_candidates.append((log_prob+np.log(probs[0][ele]), np.append(b[1], ele), np.append(b[2], dict[ele]), hout_prev, cell_in))
            
            beam_candidates.sort(key=lambda x:x[0], reverse=True)
            #beam_candidates.sort(reverse = True) # decreasing order
            beams = beam_candidates[:beam_size]
            time_stamp += 1

            if time_stamp >= max_len: break
        
        return beams[0][1], beams[0][2]
    
    """ Backward Pass """
    def bwdPass(self, dY, cache):
        Wd = cache['Wd']
        Hout = cache['Hout']
        IFOG = cache['IFOG']
        IFOGf = cache['IFOGf']
        Cellin = cache['Cellin']
        Cellout = cache['Cellout']
        Hin = cache['Hin']
        WLSTM = cache['WLSTM']
        Ws = cache['Ws']
        Ds = cache['Ds']
        Dsh = cache['Dsh']
        Wah = cache['Wah']
        feed_recurrence = cache['feed_recurrence']
        
        n,d = Hout.shape

        # backprop the hidden-output layer
        dWd = Hout.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims = True)
        dHout = dY.dot(Wd.transpose())

        # backprop the LSTM
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dCellin = np.zeros(Cellin.shape)
        dCellout = np.zeros(Cellout.shape)
        dWs = np.zeros(Ws.shape)
        
        dDsh = np.zeros(Dsh.shape)
        
        for t in reversed(xrange(n)):
            dIFOGf[t,2*d:3*d] = Cellout[t] * dHout[t]
            dCellout[t] = IFOGf[t,2*d:3*d] * dHout[t]
            
            dCellin[t] += (1-Cellout[t]**2) * dCellout[t]
            
            if t>0:
                dIFOGf[t, d:2*d] = Cellin[t-1] * dCellin[t]
                dCellin[t-1] += IFOGf[t,d:2*d] * dCellin[t]
            
            dIFOGf[t, :d] = IFOGf[t,3*d:] * dCellin[t]
            dIFOGf[t,3*d:] = IFOGf[t, :d] * dCellin[t]
            
            # backprop activation functions
            dIFOG[t, 3*d:] = (1-IFOGf[t, 3*d:]**2) * dIFOGf[t, 3*d:]
            y = IFOGf[t, :3*d]
            dIFOG[t, :3*d] = (y*(1-y)) * dIFOGf[t, :3*d]
            
            # backprop matrix multiply
            dWLSTM += np.outer(Hin[t], dIFOG[t])
            dHin[t] = dIFOG[t].dot(WLSTM.transpose())
      
            if t > 0: dHout[t-1] += dHin[t,1+Ws.shape[1]:]
            
            if feed_recurrence == 0:
                if t == 0: dDsh[t] = dIFOG[t]
            else: 
                dDsh[0] += dIFOG[t]
        
        # backprop to the diaact-hidden connections
        dWah = Ds.transpose().dot(dDsh)
        dbah = np.sum(dDsh, axis=0, keepdims = True)
             
        return {'Wah':dWah, 'bah':dbah, 'WLSTM':dWLSTM, 'Wd':dWd, 'bd':dbd}
    
    
    """ Batch data representation """
    def prepare_input_rep(self, ds, batch, params):
        batch_reps = []
        for i,x in enumerate(batch):
            batch_rep = {}
            
            vec = np.zeros((1, self.model['Wah'].shape[0]))
            vec[0][x['diaact_rep']] = 1
            for v in x['slotrep']:
                vec[0][v] = 1
            
            word_arr = x['sentence'].split(' ')
            word_vecs = np.zeros((len(word_arr), self.model['Wxh'].shape[0]))
            labels = [0] * (len(word_arr)-1)
            for w_index, w in enumerate(word_arr[:-1]):
                if w in ds.data['word_dict'].keys():
                    w_dict_index = ds.data['word_dict'][w]
                    word_vecs[w_index][w_dict_index] = 1
                
                if word_arr[w_index+1] in ds.data['word_dict'].keys():
                    labels[w_index] = ds.data['word_dict'][word_arr[w_index+1]] 
            
            batch_rep['diaact'] = vec
            batch_rep['words'] = word_vecs
            batch_rep['labels'] = labels
            batch_reps.append(batch_rep)
        return batch_reps
