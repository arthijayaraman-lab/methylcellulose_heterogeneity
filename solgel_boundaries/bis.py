import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.spatial import KDTree
from scipy.ndimage import label, zoom

import os
import json
import sys


def check_complete(dirname):
    bools = []
    file = os.path.join(dirname, 'score_arr.txt')
    if os.path.isfile(file):
        bools.append(True)
    else:
        bools.append(False)
    return np.all(bools)

def get_engdirs(batch_dir):
    dirs = []
    for item in os.listdir(batch_dir):
        if item.startswith('eng'):
            check = check_complete(os.path.join(batch_dir, item))
            if check:
                dirs.append(os.path.join(batch_dir, item))
    return dirs 

class BIS:
    def __init__(self, batch_id, bound_params):
        bp = list(bound_params.keys())
        if 'temp' in bp:
            ix = bp.index('temp')
        else:
            ix = 1
        print(bound_params.keys())

         
        if ix == 1:
            self.xparam = list(bound_params.keys())[0]
            self.yparam = list(bound_params.keys())[1]
    
            bp = list(bound_params.values())
            
            self.x_min, self.x_max = bp[0][0], bp[0][1]
            self.y_min, self.y_max = bp[1][0], bp[1][1]
            
            self.batch_id = batch_id
            self.X, self.y = self.get_init_points()
        else: #switch temp to y axis always
            self.xparam = list(bound_params.keys())[1]
            self.yparam = list(bound_params.keys())[0]
    
            bp = list(bound_params.values())
            
            self.x_min, self.x_max = bp[1][0], bp[1][1]
            self.y_min, self.y_max = bp[0][0], bp[0][1]
            
            self.batch_id = batch_id
            self.X, self.y = self.get_init_points()
            #self.X = self.X[:,::-1]
            
            
    def get_init_points(self, random_init = True, test = False):
        if test:
            N = 50
            
            if random_init:
                X = np.column_stack((np.random.uniform(self.x_min, self.x_max, N), np.random.uniform(self.y_min, self.y_max, N)))
            else:
                xg = np.linspace(self.x_min, self.x_max, int(N**0.5))
                yg = np.linspace(self.y_min, self.y_max, int(N**0.5))
                xx, yy = np.meshgrid(xg, yg)
                X = np.array([xx.ravel(), yy.ravel()]).T
    
            # Generate a boolean response for each point (0 or 1) for binary classification
            y = [0 if x[0] < 340 and x[1] < 2.5 and x[1] > 1.5 else 1 for x in X ]
        else:
            dirs = []
            for ix in range(1, self.batch_id+1):
                batch_dir = f'batch_{ix}'
                dirs.extend(get_engdirs(batch_dir))
            
            X = []
            y = []
            for d in dirs:
                xfile = os.path.join(d, 'params.json')
                yfile = os.path.join(d, 'score_arr.txt')
                
                
                with open(xfile, 'r') as f:
                    xparams = json.load(f)
                X.append([float(xparams[self.xparam]), float(xparams[self.yparam])])
                y.append(np.loadtxt(yfile))
            X = np.array(X)
            y = np.array(y)
        return X, y
    
    def get_boundary(self, plot_bound_w_pts = False, plot_bound= False, ax = None, boot = False):
        Ngrid = 100

        if np.all(self.y == self.y[0]):
            self.boundary = boundary = np.zeros((Ngrid, Ngrid)).astype(bool)
            
            if plot_bound_w_pts:
                if not ax:
                    f, ax = plt.subplots(figsize=(4, 4))
                ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=50, cmap='coolwarm', edgecolors='k', vmin = 0, vmax = 1)
                ax.set_ylim(self.y_min, self.y_max)
                ax.set_xlim(self.x_min, self.x_max)
                ax.set_ylabel(self.yparam)
                ax.set_xlabel(self.xparam)
            
        else:
        
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel="rbf", probability=True, C = 200))
            ])
            
            # Set up parameter grid for gamma values to test
            param_grid = {
                'svc__gamma': ['scale']#np.logspace(-3, 2, 10)  # Range of gamma values from 0.001 to 100
            }
            
            # Use KFold cross-validation to select the optimal gamma
            cv = KFold(n_splits=5, shuffle=True)#, random_state=0)
            
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')
            
            if boot:
                ndata = len(self.X)
                ixs = np.random.choice(np.array(range(ndata)), ndata)
                X = self.X[ixs]
                y = self.y[ixs]
            else:
                X = self.X
                y = self.y
                
            grid_search.fit(X, y)
            
            # Get the best model from cross-validation
            best_model = grid_search.best_estimator_
            self._svc = best_model
    
            #lists = [[np.min(X[:,i]), np.max(X[:,i])] for i in range(X.shape[1])]
            self.xl = xl = np.linspace(self.x_min, self.x_max, Ngrid)
            self.yl = yl = np.linspace(self.y_min, self.y_max, Ngrid)
            xx, yy = np.meshgrid(xl,yl )
            pts = np.c_[xx.ravel(), yy.ravel()]
            Z = best_model.predict(pts)
            Z = Z.reshape(xx.shape)
            grad = np.gradient(Z.T)
            boundary = np.sum(np.abs(grad), 0) > 0
            self.boundary = boundary.T

            if plot_bound_w_pts:
                if not ax:
                    f, ax = plt.subplots(figsize=(4, 4))
                
                # Plot the boundary and the points
                ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
                ax.contour(xx, yy, Z, colors = 'k', levels = [0.5])
                ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=50, cmap='coolwarm', edgecolors='k')
                ax.set_ylim(self.y_min, self.y_max)
                ax.set_xlim(self.x_min, self.x_max)
                ax.set_ylabel(self.yparam)
                ax.set_xlabel(self.xparam)
                #ax.set_title(f'Optimal Boundary with RBF Kernel (Best gamma = {grid_search.best_params_["svc__gamma"]:.4f})')
                
                
                #plt.figure()
                #plt.title('boundary')
                
            if plot_bound:
                if not ax:
                    f, ax = plt.subplots(figsize=(4, 4))
                    
                # Plot the boundary and the points
                ax.contour(xx, yy, Z, colors = 'k', levels = [0.5])
                ax.set_ylim(self.y_min, self.y_max)
                ax.set_xlim(self.x_min, self.x_max)
                ax.set_ylabel(self.yparam)
                ax.set_xlabel(self.xparam)
                #ax.set_title(f'Optimal Boundary with RBF Kernel (Best gamma = {grid_search.best_params_["svc__gamma"]:.4f})')
                

        return boundary
    
    def select_spaced_points(self, boolean_array, N, min_distance):
        # Get indices of True values
        valid_points = np.argwhere(boolean_array)
        selected_points = []
        
        while len(selected_points) < N and len(valid_points) > 0:
            if len(selected_points) == 0:
                # Randomly select the first point
                selected_idx = np.random.choice(len(valid_points))
            else:
                # Compute distances from all valid points to all selected points
                distances = np.linalg.norm(valid_points[:, None] - np.array(selected_points), axis=2)
                # Find points that are farther than min_distance from all selected points
                valid_mask = (distances > min_distance).all(axis=1)
                valid_points = valid_points[valid_mask]
                
                if len(valid_points) == 0:
                    break  # No more points meet the criteria
                
                selected_idx = np.random.choice(len(valid_points))
            
            # Add the selected point
            selected_points.append(valid_points[selected_idx])
            valid_points = np.delete(valid_points, selected_idx, axis=0)
        
        return np.array(selected_points)

    def get_spaced_points(self, N):
        min_distance = 100
        run = True
        while run:
            pts = self.select_spaced_points(self.boundary.T, N, min_distance)
            if len(pts) == N:
                return pts
            min_distance -=1
    
    def get_new_points(self, N, r = 1, fmt = True):
        if self.batch_id < 5:
            ret_pts = np.column_stack((np.random.uniform(self.x_min, self.x_max, N), np.random.uniform(self.y_min, self.y_max, N)))
            if fmt:
                print('formatting points')
                ret_pts = self.format_pts(ret_pts)
            return ret_pts
        
        self.get_boundary()
        
        if np.sum(self.boundary) ==0: # if no boundary, return random points
            ret_pts = np.column_stack((np.random.uniform(self.x_min, self.x_max, N), np.random.uniform(self.y_min, self.y_max, N)))
            if fmt:
                print('formatting points')
                ret_pts = self.format_pts(ret_pts)
            return ret_pts
        
        spaced_pts = self.get_spaced_points(3*N)
        
        x = self.xl[spaced_pts[:,0]]
        y = self.yl[spaced_pts[:,1]]
        boundary_pts = np.stack((x, y), -1)
        
        tree = KDTree(self.X)
        dists = []
        for pt in boundary_pts:
            d, _ = tree.query(pt, k = 1)
            dists.append(d)
        weights = np.array(dists)**2
        probs = weights/np.sum(weights)
        inds = list(range(len(boundary_pts)))
        sel_inds = np.random.choice(inds, N, replace = False, p = probs)
        ret_pts = boundary_pts[sel_inds]

        if fmt:
            print('formatting points')
            ret_pts = self.format_pts(ret_pts)
        return ret_pts
        
    def format_pts(self, pts):
        params = ['temp', 'ds', 'H', 'dp', 'nchains', 'conc']
        ret_pts = np.zeros((len(pts), len(params)))
        
        opfile = 'chain_params.json'
        with open(opfile, 'r') as f:
            other_params = json.load(f)
            
        opk = list(other_params.keys())
        for i, p in enumerate(params):
            if p == self.xparam:
                ret_pts[:,i] = pts[:,0]
            elif p == self.yparam:
                ret_pts[:,i] = pts[:,1]
            else:
                assert p in opk
                ret_pts[:,i] = other_params[p]
        return ret_pts
    
    def get_probs(self, N = 1):
        ap = []
        if N > 1: boot = True
        else: boot = False
        for i in range(N):
            self.get_boundary(boot = boot)
            xx, yy = np.meshgrid(self.xl,self.yl )
            pts = np.c_[xx.ravel(), yy.ravel()]
            Z = self._svc.predict_proba(pts, )[:,0]
            Z = Z.reshape(xx.shape)
            ap.append(Z)
        mean = np.mean(ap, 0)
        return mean, ap
    
    def plot_connected(self, p, contour, lev, color, xl, yl):
        x = xl #self.xl
        y = yl #self.yl
        levels_regions =[lev, 1-lev]
        
        curve_coords = []
        for collection in contour.collections:
            for path in collection.get_paths():
                verts = path.vertices
                curve_coords.append(verts)
                
        binary_mask = np.logical_and(p > lev, p<1-lev)
        labeled_mask, num_features = label(binary_mask)
        
        def is_component_connected(component_mask, curve_coords):
            """Check if a labeled component is connected to the curve."""
            for verts in curve_coords:
                curve_x, curve_y = verts[:, 0], verts[:, 1]
                # Find the indices corresponding to curve points
                xi = np.round((curve_x - x[0]) / (x[1] - x[0])).astype(int)
                yi = np.round((curve_y - y[0]) / (y[1] - y[0])).astype(int)
                xi = np.clip(xi, 0, len(x) - 1)
                yi = np.clip(yi, 0, len(y) - 1)

                # Check if any curve point touches the component
                if np.any(component_mask[yi, xi]):
                    return True
            return False
        
        connected_mask = np.zeros_like(labeled_mask, dtype=bool)
        for i in range(1, num_features + 1):  # Exclude background (label 0)
            component_mask = labeled_mask == i
            if is_component_connected(component_mask, curve_coords):
                connected_mask |= component_mask
        
        p[np.where(np.logical_and(p < 0.5, ~connected_mask))] = levels_regions[0]-0.01
        p[np.where(np.logical_and(p > 0.5, ~connected_mask))] = levels_regions[1]+0.01
        ci = plt.contourf(xl, yl, p, levels=levels_regions, colors =  [color], alpha = 0.2)
        
        return ci
    
    def plot_bound_w_err(self, Nboot = 1, lev = 0.025, ax = None, color = 'k', exclude_nonconnected = True):
        if not ax:
            f, ax = plt.subplots()

        p, ap = self.get_probs(Nboot)
        smooth = 5
        p = zoom(p, smooth)
        xl = np.linspace(min(self.xl), max(self.xl), smooth * len(self.xl))
        yl = np.linspace(min(self.yl), max(self.yl), smooth * len(self.yl))

        bound = ax.contour(
                xl,
                yl,
                p,
                colors=[color],
                levels=[0.5],
                alpha=1,)
        if exclude_nonconnected:
            ci = self.plot_connected(p, bound, lev, color, xl, yl)
        else:
            ci = ax.contourf(
                        xl,
                        yl,
                        p,
                        colors=[color],
                        levels=[lev, 1-lev],
                        alpha=0.2,
                    )
        
        
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylabel(self.yparam)
        ax.set_xlabel(self.xparam)
        return ax, p, ap
    
    def check_done(self, b1, b2, thresh = 1, plot = True):
        b1_pts = np.array(np.where(b1)).T
        b2_pts = np.array(np.where(b2)).T
        tree = KDTree(b2_pts)
        d, n = tree.query(b1_pts, k = 1)
        mae = np.mean(d)
        print(mae)
        return mae < thresh
    
    def write_boundary_file(self, batch_no):
        np.savetxt(os.path.join(f'batch_{batch_no}', 'boundary.txt'), self.boundary)
        
    def load_boundary_file(self, batch_no):
        return np.loadtxt(os.path.join(f'batch_{batch_no}', 'boundary.txt'))
    
    def check_N_batches(self, N = 5):
        #run at the start of a batch and will end if previous batches converge
        stop = []
        if N >= self.batch_id:
            return 0
        for i in range(N):
            batch_no = (self.batch_id-N)+i
            b1 = self.load_boundary_file(batch_no)
            b2 = self.load_boundary_file(batch_no+1)
            
            check = self.check_done(b1, b2)
            stop.append(check)
            
        if np.all(stop): 
            self.stop_bis()
            
    def stop_bis():
        sys.exit("Ending BIS with successful convergence")
        
"""Roughly BROAD INITIAL SAMPLING (BIS) from this paper:
    https://pubs.acs.org/doi/10.1021/acs.iecr.3c02362"""
    
def test():
    # bound_params = {'temp':[298, 348],
    #                 'ds': [0,3]}
    
    with open('bound_params.json','r') as f:
        bound_params = json.load(f)
    print(bound_params)
    
    bis = BIS(0, bound_params)
    bis.get_boundary(plot_bound_w_pts = True)
    x = bis.get_new_points(10, fmt = True)
    #plt.scatter(x[:,0], x[:,1], color = 'k', marker = 'x')
    return bis, x

if __name__ == '__main__':
    bis, x = test()
    #oldtest()
    



