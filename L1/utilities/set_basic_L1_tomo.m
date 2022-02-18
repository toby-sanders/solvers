% some default L1 optimization parameters for SAR

clear pat;

pat.mu = 500;
pat.mu0 = pat.mu;
pat.order = 1;
pat.levels = 1;
pat.beta = 2^5;
pat.outer_iter = 15;
pat.inner_iter = 10;
pat.data_mlp = false;


pat.nonneg = false;
pat.isreal = true;
pat.max_c = false;
pat.max_v = 0;
pat.disp = 1;
pat.wrap_shrink = false;

