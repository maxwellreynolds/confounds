# WHere we are at: 7/7/22:
# need to do transform, and put the subject info into self as is done with the batch info.
# The transform should be fairly straightforward just add 

import numpy as np
import pdb
from sklearn.utils.validation import (check_array, check_consistent_length,
                                      check_is_fitted, column_or_1d)
from confounds.base import BaseDeconfound
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
#export PYTHONPATH="${PYTHONPATH}:/Users/maxreynolds/Desktop/ConfoundsRepo/"

class LongComBat(BaseDeconfound):
    """ComBat method to remove batch effects."""

    def __init__(self,
                 # parametric=True, # TODO: When implmenented non-parametric
                 # adjust_variance=True, # TODO: When implmented only mean
                 tol=1e-4):
        """Initiate object."""
        super().__init__(name='LongComBat')
        # self.parametric = True
        # self.adjust_variance = True
        self.tol = tol

    def fit(self,
            in_features,
            batch,
            subject,
            effects_interest=None
            ):
        """
        Fit Combat.

        Estimate parameters in the Combat model. This operation will estimate
        the scale and location effects in the batches supplied, and the
        coefficients for the effects to keep after harmonisation.

        Parameters
        ----------
        in_features : {array-like, sparse matrix},
                      shape (n_samples, n_features)
            The training input samples.
        batch : ndarray, shape (n_samples, )
            Array of batches.
        subject : identifier for the subject (n_samples, )
            Array of subjects
        effects_interest: ndarray, shape (n_samples, n_features_of_effects),
            optinal.
            Array of effects of interest to keep after harmonisation.

        Returns
        -------
        self: returns an instance of self.

        """

        in_features = check_array(in_features)
        # pdb.set_trace()
        batch = column_or_1d(batch)
        # pdb.set_trace()
        subject = column_or_1d(subject)

        if effects_interest is not None:
            effects_interest = check_array(effects_interest)

        check_consistent_length([in_features,
                                 batch,
                                 effects_interest,
                                 subject])

        return self._fit(Y=in_features,
                         b=batch,
                         X=effects_interest,
                         s=subject
                         )

    def _fit(self, Y, b, X, s):
        """max implementation"""
        # b_rename=np.array(['batch_'+str(b_elem) for b_elem in b])
        b_rename=np.array([str(b_elem) for b_elem in b])
        s_rename=np.array([str(s_elem) for s_elem in s])
        V = Y.shape[1]
        # pdb.set_trace()
        print(V,'features')
        R = Y.shape[0]
        print(R, 'measurements')

        predicted = []
        # pdb.set_trace()

        batch = np.unique(b_rename)
        batch.sort()

        subject = np.unique(s_rename)
        
        batches = [] #index for every observation in a scanner group

        for b_elem in batch:
            # batches.append(list(df[df['Scanner_Proxy'] == s].index))
            batches.append(list(np.where(b_rename==b_elem)[0]))

        m = len(batches)
        ni = np.array([len(x) for x in batches])
        L = Y.shape[0]
        # pdb.set_trace()
        df=pd.DataFrame(data=np.concatenate((X,Y),axis=1),columns=['x'+str(i) for i in range(X.shape[1])]+['y'+str(i) for i in range(Y.shape[1])])
        df['batch']=np.expand_dims(b_rename,1)
        # df[['x'+str(i) for i in range(X.shape[1])]+['y'+str(i) for i in range(Y.shape[1])]]=df[['x'+str(i) for i in range(X.shape[1])]+['y'+str(i) for i in range(Y.shape[1])]].astype(np.float)
        batch_effects = []
        X_coeffs = []
        predicted = []
        sigma_estimates = []
        intercepts=[]
        # pdb.set_trace()
        eta_estimates = []
        for i in range(V):
            print(i)
            term_str=' + '.join(['x'+str(i) for i in range(X.shape[1])]+['batch'])
            formula = "{} ~ {}".format('y'+str(i),term_str)
            print(formula)
            md = smf.mixedlm(formula, df, groups = s_rename)
            mdf = md.fit(reml = True)
            intercepts.append(mdf.fe_params.Intercept)

            batchef = mdf.fe_params.filter(regex = 'batch') 
            pred = mdf.fittedvalues
            sig = np.sqrt(mdf.scale) #save standard deviation of residuals 
            X_coeff=mdf.fe_params[['x'+str(i) for i in range(X.shape[1])]]
            batch_effects.append(batchef) #save scanner effects
            X_coeffs.append(X_coeff)
            predicted.append(pred)
            sigma_estimates.append(sig)

            eta_estimates.append([mdf.random_effects[s_elem]['Group'] for s_elem in subject])
            
            if i % 10 == 0:
                print(i+1, 'of', V, 'features done')

        eta_estimates = np.array(eta_estimates).T
        self.eta_estimates_ = eta_estimates
        pdb.set_trace()
        batch_effects = np.array(batch_effects).T
        X_coeffs = np.array(X_coeffs).T
        intercepts = np.array(intercepts)
        sigmas = np.tile(np.array(sigma_estimates), (df.shape[0],1))

        #calculate gamma1hats
        #weighted sum of scanner effects for each feature (weighted by #obs in feature)
        gamma1hat = -np.matmul(np.expand_dims(ni[1:], 0), batch_effects)/L

        batch_effects_adjusted = batch_effects + np.tile(gamma1hat, (batch_effects.shape[0],1))
        batch_effects_adjusted = np.concatenate((gamma1hat, batch_effects_adjusted), axis=0)

        batch_effects_expanded = np.zeros((L,V))

        for i in range(m):
            batch_effects_expanded[batches[i]] = np.tile(batch_effects_adjusted[i], (len(batches[i]),1))

        pdb.set_trace()
        epsilon = np.mean((df[['y'+str(i) for i in range(Y.shape[1])]].values-np.array(predicted).T)**2,axis=0)

        data_std= (df[['y'+str(i) for i in range(Y.shape[1])]] - np.array(predicted).T + batch_effects_expanded) / sigmas
        
        gammahat = np.zeros((m, V))
        delta2hat = np.zeros((m, V))
        for i in range(m): #for every scanner
            gammahat[i] = data_std.values[batches[i],].mean(axis=0)
            delta2hat[i] = np.var(data_std.values[batches[i],], axis=0, ddof=1)
        
        gammabar = gammahat.mean(axis = 1)
        tau2bar = gammahat.var(axis = 1, ddof=1)

        dbar = delta2hat.mean(axis = 1)
        S2bar = delta2hat.var(axis = 1, ddof=1)
        lambdabar = (dbar**2 + 2*S2bar)/S2bar
        thetabar = (dbar**3 + dbar*S2bar)/S2bar

        print('ni',ni.shape)
        print('tau2bar',tau2bar.shape)
        print('gammabar',gammabar.shape)
        print('gammahat',gammahat.shape)
        print('delta2hat',gammahat.shape)

        gammastarhat0 = (( np.tile(ni,(V,1)).T* np.tile(tau2bar,(V,1)).T *gammahat) + (delta2hat*np.tile(gammabar,(V,1)).T)) / ((np.tile(ni,(V,1)).T* np.tile(tau2bar,(V,1)).T) + delta2hat)
        gammastarhat_mu_0=((np.tile(ni,(V,1)).T*np.tile(tau2bar,(V,1)).T*gammahat) +(delta2hat*np.tile(gammabar,(V,1)).T)) / (((np.tile(ni,(V,1)).T*np.tile(tau2bar,(V,1)).T) + delta2hat))
        gammastarhat_sigma_0=1/((np.tile(ni,(V,1)).T*np.tile(tau2bar,(V,1)).T) + delta2hat)/(delta2hat*np.tile(tau2bar,(V,1)).T)

        delta2starhat0=np.zeros((m,V))
        delta2starhat_alpha0=(np.tile(ni,(V,1)).T/2 + np.tile(lambdabar,(V,1)).T)
        delta2starhat_beta0=np.zeros((m,V))
        for i in range(m):
            zminusgammastarhat2 = ((data_std.loc[batches[i]].values - gammastarhat0[i,])**2).sum(axis=0)
            delta2starhat0[i] = (thetabar[i]+0.5*zminusgammastarhat2) / (ni[i]/2+lambdabar[i]-1)
            delta2starhat_beta0[i] = thetabar[i]+0.5*zminusgammastarhat2

        niter=100


        gammastarhat=np.zeros((niter,m,V,))
        gammastarhat[0,:,:]=gammastarhat0
        gammastarhat_mus=np.zeros((niter,m,V,))
        gammastarhat_mus[0,:,:]=gammastarhat_mu_0
        gammastarhat_sigmas=np.zeros((niter,m,V,))
        gammastarhat_sigmas[0,:,:]=gammastarhat_sigma_0
        delta2starhat=np.zeros((niter,m,V))
        delta2starhat[0,:,:]=delta2starhat0
        delta2starhat_alphas=np.zeros((niter,m,V,))
        delta2starhat_alphas[0,:,:]=delta2starhat_alpha0
        delta2starhat_betas=np.zeros((niter,m,V,))
        delta2starhat_betas[0,:,:]=delta2starhat_beta0

        for b in range(1,niter): 
            gammastarhat_mu=((np.tile(ni,(V,1)).T*np.tile(tau2bar,(V,1)).T*gammahat) +(delta2starhat[b-1,:,:]*np.tile(gammabar,(V,1)).T)) / (((np.tile(ni,(V,1)).T*np.tile(tau2bar,(V,1)).T) + delta2starhat[b-1,:,:]))
            gammastarhat_sigma=1/((np.tile(ni,(V,1)).T*np.tile(tau2bar,(V,1)).T) + delta2starhat[b-1,:,:])/(delta2starhat[b-1,:,:]*np.tile(tau2bar,(V,1)).T)
        #     gammastarhat[b,:,:]=dist.Normal(torch.tensor(gammastarhat_mu), torch.tensor(np.sqrt(gammastarhat_sigma))).sample()
            gammastarhat[b,:,:]=gammastarhat_mu
            
            delta2starhat_alpha=(np.tile(ni,(V,1)).T/2 + np.tile(lambdabar,(V,1)).T)
            delta2starhat_beta=np.zeros((m,V))
        #     pdb.set_trace()
            for i in range(m):
                zminusgammastarhat2 = ((data_std.loc[batches[i]].values - gammastarhat[b-1,i,:])**2).sum(axis=0)
                delta2starhat_beta[i,:]= thetabar[i]+0.5*zminusgammastarhat2
        #     delta2starhat[b,:,:]=dist.InverseGamma(torch.tensor(delta2starhat_alpha),torch.tensor(delta2starhat_beta)).sample()
                delta2starhat[b,:,:]=(delta2starhat_beta)/(delta2starhat_alpha-1)
            
            delta2starhat_alphas[b,:,:]=delta2starhat_alpha
            delta2starhat_betas[b,:,:]=delta2starhat_beta
            gammastarhat_mus[b,:,:]=gammastarhat_mu
            gammastarhat_sigmas[b,:,:]=gammastarhat_sigma
            
            if b % 10 == 9:
                print(b+1, ' samples done')
        
        gammastarhat_final=gammastarhat[-1,:,:]
        delta2starhat_final=delta2starhat[-1,:,:]
        gammastarhat_mu_final=gammastarhat_mus[-1,:,:]
        gammastarhat_sigma_final=gammastarhat_sigmas[-1,:,:]
        delta2starhat_alpha_final=delta2starhat_alphas[-1,:,:]
        delta2starhat_beta_final=delta2starhat_betas[-1,:,:]
        print(delta2starhat_beta_final.shape)

        gammastarhat_expanded = np.zeros((L,V))
        
        delta2starhat_expanded = np.zeros((L,V))

        for i in range(m):
            gammastarhat_expanded[batches[i]]=np.tile(gammastarhat_final[i,],(len(batches[i]),1))
            delta2starhat_expanded[batches[i]]=np.tile(delta2starhat_final[i,],(len(batches[i]),1))

        self.gamma_ = gammastarhat_final
        self.delta_sq_ = delta2starhat_final
        self.epsilon_ = epsilon
        self.intercept_ = intercepts
        self.coefs_x_ = X_coeffs
        self.batches_ = batch
        self.subjects_ = subject
        
        """end max implementation"""
        """Actual fit method."""
        # # extract unique batch categories
        # batches = np.unique(b)
        # self.batches_ = batches

        # # Construct one-hot-encoding matrix for batches
        # B = np.column_stack([(b == b_name).astype(int)
        #                      for b_name in self.batches_])

        # n_samples, n_features = Y.shape
        # n_batch = B.shape[1]

        # if n_batch == 1:
        #     raise ValueError('The number of batches should be at least 2')

        # sample_per_batch = B.sum(axis=0)

        # if np.any(sample_per_batch == 1):
        #     raise ValueError('Each batch should have at least 2 observations'
        #                      'In the future, when this does not happens,'
        #                      'only mean adjustment will take place')

        # # Construct design matrix
        # M = B.copy()
        # if isinstance(X, np.ndarray):
        #     M = np.column_stack((M, X))
        #     end_x = n_batch + X.shape[1]
        # else:
        #     end_x = n_batch

        # # OLS estimation for standardization
        # beta_hat = np.matmul(np.linalg.inv(np.matmul(M.T, M)),
        #                      np.matmul(M.T, Y))

        # # Find grand mean intercepts, from batch intercepts
        # alpha_hat = np.matmul(sample_per_batch/float(n_samples),
        #                       beta_hat[:n_batch, :])
        # self.intercept_ = alpha_hat

        # # Find slopes for the  effects of interest
        # coefs_x = beta_hat[n_batch:end_x, :]
        # self.coefs_x_ = coefs_x

        # # Compute error between predictions and observed values
        # Y_hat = np.matmul(M, beta_hat)  # fitted observations
        # epsilon = np.mean(((Y - Y_hat)**2), axis=0)
        # self.epsilon_ = epsilon

        # # Standardise data
        # Z = Y.copy()
        # Z -= alpha_hat[np.newaxis, :]
        # Z -= np.matmul(M[:, n_batch:end_x], coefs_x)
        # Z /= np.sqrt(epsilon)

        # # Find gamma fitted to Standardised data
        # gamma_hat = np.matmul(np.linalg.inv(np.matmul(B.T, B)),
        #                       np.matmul(B.T, Z)
        #                       )
        # # Mean across input features
        # gamma_bar = np.mean(gamma_hat, axis=1)
        # # Variance across input features

        # if n_features > 1:
        #     ddof_feat = 1
        # else:
        #     raise print("Dataset with just one feature will give NaNs when "
        #                 "computing the variance across features. This will "
        #                 "be fixed in the feature")
        #     # ddof_feat = 0
        # tau_bar_sq = np.var(gamma_hat, axis=1, ddof=ddof_feat)
        # # tau_bar_sq += 1e-10

        # # Variance per batch and gen
        # delta_hat_sq = [np.var(Z[B[:, ii] == 1, :], axis=0, ddof=1)
        #                 for ii in range(B.shape[1])]
        # delta_hat_sq = np.array(delta_hat_sq)

        # # Compute inverse moments
        # lamba_bar = np.apply_along_axis(self._compute_lambda,
        #                                 arr=delta_hat_sq,
        #                                 axis=1,
        #                                 ddof=ddof_feat)
        # thetha_bar = np.apply_along_axis(self._compute_theta,
        #                                  arr=delta_hat_sq,
        #                                  axis=1,
        #                                  ddof=ddof_feat)

        # # if self.parametric: # TODO: Uncomment when implemented
        # #     it_eb = self._it_eb_param
        # # else:
        # #     it_eb = self._it_eb_non_param

        # it_eb = self._it_eb_param
        # gamma_star, delta_sq_star = [], []
        # for ii in range(B.shape[1]):
        #     g, d_sq = it_eb(Z[B[:, ii] == 1, :],
        #                     gamma_hat[ii, :],
        #                     delta_hat_sq[ii, :],
        #                     gamma_bar[ii],
        #                     tau_bar_sq[ii],
        #                     lamba_bar[ii],
        #                     thetha_bar[ii],
        #                     self.tol
        #                     )

        #     gamma_star.append(g)
        #     delta_sq_star.append(d_sq)

        # gamma_star = np.array(gamma_star)
        # delta_sq_star = np.array(delta_sq_star)

        # self.gamma_ = gamma_star
        # self.delta_sq_ = delta_sq_star

        # print("self.gamma_", self.gamma_.shape)
        # print("self.delta_sq_", self.delta_sq_.shape)
        # print("self.epsilon_",self.epsilon_.shape)
        # print("self.intercept_",self.intercept_.shape)
        # print("self.coefs_x_",self.coefs_x_.shape)
        return self

    def transform(self,
                  in_features,
                  batch,
                  subject,
                  effects_interest=None):
        """
        Harmonise input features using an already estimated Combat model.

        Parameters
        ----------
        in_features : {array-like, sparse matrix},
                      shape (n_samples, n_features)
            The training input samples.
        batch : ndarray, shape (n_samples, )
            Array of batches.
        effects_interest: ndarray, shape (n_samples, n_features_of_effects),
            optinal.
            Array of effects of interest to keep after harmonisation.

        Returns
        -------
        in_features_transformed : harmonised in_features
        """

        in_features, batch, effects_interest = self._validate_for_transform(
            in_features, batch, effects_interest)

        return self._transform(in_features,
                               batch,
                               subject,
                               effects_interest)

    def _transform(self, Y, b, s, X):
        """Max implementation"""
        
        
        """Actual deconfounding of the test features."""
        # test_batches = np.unique(b)
        b_rename = np.array(str(b_elem) for b_elem in b)
        test_batches = np.array([str(b_elem) for b_elem in np.unique(b)])
        s_rename = np.array([str(s_elem) for s_elem in s])
        test_subjects = np.unique(s_rename)

        # First standarise again the data
        Y_trans = Y - self.intercept_[np.newaxis, :]

        if self.coefs_x_.size > 0:
            Y_trans -= np.matmul(X, self.coefs_x_)

        ##Change this to subjects
        for subject in test_subjects:
            # pdb.set_trace()
            ix_subject = np.where(self.subjects_ == subject)[0][0]
            
            Y_trans[s_rename == subject, :] -= self.eta_estimates_[ix_subject] #subject effect... then add back later
        pdb.set_trace()
        ##############

        Y_trans /= np.sqrt(self.epsilon_)

        for batch in test_batches:

            ix_batch = np.where(self.batches_ == batch)[0]

            Y_trans[b_rename == batch, :] -= self.gamma_[ix_batch]
            Y_trans[b_rename == batch, :] /= np.sqrt(self.delta_sq_[ix_batch, :])
        Y_trans *= np.sqrt(self.epsilon_)

        # Add intercept
        Y_trans += self.intercept_[np.newaxis, :]

        # Add effects of interest, if there's any
        if self.coefs_x_.size > 0:
            Y_trans += np.matmul(X, self.coefs_x_)

        for subject in test_subjects:
            # pdb.set_trace()
            ix_subject = np.where(self.subjects_ == subject)[0][0]
            Y_trans[s_rename == subject, :] += self.eta_estimates_[ix_subject]

        pdb.set_trace() 
        return Y_trans

    def _validate_for_transform(self, Y, b, X):

        # check if fitted
        attributes = ['intercept_', 'coefs_x_', 'epsilon_',
                      'gamma_', 'delta_sq_']

        # Check if Combat was previously fitted
        check_is_fitted(self, attributes=attributes)

        # Ensure that data are numpy array objects
        Y = check_array(Y)
        if X is not None:
            X = check_array(X)

        # Check that input arrays have the same observations
        check_consistent_length([Y, b, X])

        if Y.shape[1] != len(self.intercept_):
            raise ValueError("Wrong number of features for Y")

        # Check that supplied batches exist in the fitted object
        b_not_in_model = np.in1d(np.unique(b), self.batches_, invert=True)
        if np.any(b_not_in_model):
            raise ValueError("test batches categories not in "
                             "the trained model")

        if self.coefs_x_.size > 0:
            if X is None:
                raise ValueError("Effects of interest should be supplied, "
                                 "since Combat was fitted with them")
            if X.shape[1] != self.coefs_x_.shape[0]:
                raise ValueError("Dimensions of fitted beta "
                                 "and input X matrix do not match")

        return Y, b, X

    def fit_transform(self,
                      in_features,
                      batch,
                      subject,
                      effects_interest=None):
        """
        Concatenate fit and transform operations.

        Fit combat and then transform on the same data. You may want
        to use this function for training data harmonisation

       Parameters
        ----------
        in_features : {array-like, sparse matrix},
                      shape (n_samples, n_features)
            The training input samples.
        batch : ndarray, shape (n_samples, )
            Array of batches.
        subject : identifier for the subject (n_samples, )
            Array of subjects
        effects_interest: ndarray, shape (n_samples, n_features_of_effects),
            optinal.
            Array of effects of interest to keep after harmonisation.

        Returns
        -------
        in_features_transformed : harmonised in_features

        """
        # Fit Combat
        self.fit(in_features=in_features,
                 batch=batch,
                 subject=subject,
                 effects_interest=effects_interest
                 )
        # Use same data to harmonise it
        return self.transform(in_features=in_features,
                              batch=batch,
                              subject=subject,
                              effects_interest=effects_interest)

    def _it_eb_param(self,
                     Z_batch,
                     gam_hat_batch,
                     del_hat_sq_batch,
                     gam_bar_batch,
                     tau_sq_batch,
                     lam_bar_batch,
                     the_bar_batch,
                     conv):
        """Parametric EB estimation of location and scale paramaters."""
        # Number of non nan samples within the batch for each variable
        n = np.sum(1 - np.isnan(Z_batch), axis=0)
        gam_prior = gam_hat_batch.copy()
        del_sq_prior = del_hat_sq_batch.copy()

        change = 1
        count = 0
        while change > conv:
            gam_post = self._post_gamma(del_sq_prior,
                                        gam_hat_batch,
                                        gam_bar_batch,
                                        tau_sq_batch,
                                        n)

            del_sq_post = self._post_delta(gam_post,
                                           Z_batch,
                                           lam_bar_batch,
                                           the_bar_batch,
                                           n)

            change = max((abs(gam_post - gam_prior) / gam_prior).max(),
                         (abs(del_sq_post - del_sq_prior) / del_sq_prior).max()
                         )
            gam_prior = gam_post
            del_sq_prior = del_sq_post
            count = count + 1

        # TODO: Make namedtuple?
        return (gam_post, del_sq_post)

    def _it_eb_non_param():
        # TODO
        return NotImplementedError()

    @staticmethod
    def _compute_lambda(del_hat_sq, ddof):
        """Estimation of hyper-parameter lambda."""
        v = np.mean(del_hat_sq)
        s2 = np.var(del_hat_sq, ddof=ddof)
        # s2 += 1e-10
        # In Johnson 2007  there's a typo
        # in the suppl. material as it
        # should be with v^2 and not v
        return (2*s2 + v**2)/float(s2)

    @staticmethod
    def _compute_theta(del_hat_sq, ddof):
        """Estimation of hyper-parameter theta."""
        v = del_hat_sq.mean()
        s2 = np.var(del_hat_sq, ddof=ddof)
        # s2 += 1e-10
        return (v*s2+v**3)/s2

    @staticmethod
    def _post_gamma(x, gam_hat, gam_bar, tau_bar_sq, n):
        # x is delta_star
        num = tau_bar_sq*n*gam_hat + x * gam_bar
        den = tau_bar_sq*n + x
        return num/den

    @staticmethod
    def _post_delta(x, Z, lam_bar, the_bar, n):
        num = the_bar + 0.5*np.sum((Z - x[np.newaxis, :])**2, axis=0)
        den = n/2.0 + lam_bar - 1
        return num/den
