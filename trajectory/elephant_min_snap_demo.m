%% elephant_min_snap_demo
% Minimum-snap elephant through 15 waypoints (the path flown in the final runs).
% Frame: metres from the start point, x forward, y right; the drone keeps its take-off heading.
% One 7th-order polynomial per leg; position passes through every waypoint, and velocity,
% acceleration and jerk are continuous. Output: elephant_min_snap_traj.csv (t, x, y, z).

clc; close all;
traj_dir = fileparts(mfilename('fullpath'));

% Waypoints in order (cm; x forward, y right). The path starts and lands at (0, 0).
pts_cm = [  0    0;
          140  -90;
          200  -90;
          320    0;
          380  100;
          250  120;
          140  100;
           80  -20;
          140 -140;
          250 -160;
          380 -140;
          320    0;
          200   65;
          140   65;
            0    0];

% Parameters
max_leg_m   = 0;       % 0 = pass through the waypoints only (no extra guide points)
v_limit     = 0.297;   % peak speed (m/s); the mission script refuses > 0.30 m/s
end_factor  = 1.5;     % first and last legs get more time, so the drone starts and stops at rest
balance_its = 15;      % passes that even out the peak speed of each leg
dt          = 0.01;    % output sample time (s)

% Waypoints in metres, plus optional guide points along each leg
P_req = pts_cm / 100;
W = P_req(1, :);
pt_idx = 1;
for k = 1:size(P_req, 1) - 1
    seg = P_req(k+1, :) - P_req(k, :);
    n = 1;
    if max_leg_m > 0, n = max(1, ceil(norm(seg) / max_leg_m)); end
    W = [W; P_req(k, :) + (1:n).' / n * seg];
    W(end, :) = P_req(k+1, :);
    pt_idx(end+1) = size(W, 1);
end
M = size(W, 1) - 1;

% Leg times: proportional to leg length, balanced, then scaled so the peak speed is v_limit
T = vecnorm(diff(W), 2, 2).' / v_limit;
T([1 M]) = T([1 M]) * end_factor;
for it = 1:balance_its
    coeff = solveMinSnap(W, T);
    pk = legPeaks(coeff, T);
    T = T .* sqrt(pk / mean(pk));
end
coeff = solveMinSnap(W, T);
T = T * max(legPeaks(coeff, T)) / v_limit;
coeff = solveMinSnap(W, T);

% Sample the path and rescale time until the sampled peak speed is within the limit
for pass = 1:3
    t = (0:dt:sum(T)).';
    if t(end) < sum(T), t(end+1) = sum(T); end
    [P, V] = sampleMinSnap(coeff, T, t);
    speed = vecnorm(V, 2, 2);
    if max(speed) <= v_limit, break; end
    T = T * max(speed) / v_limit * 1.0005;
    coeff = solveMinSnap(W, T);
end
P(end, :) = P_req(end, :);

% Report path length, extent, duration and speed
fprintf('%d points (+ %d guide points), %.2f m of path\n', size(P_req, 1), ...
    size(W, 1) - size(P_req, 1), sum(speed(1:end-1) .* diff(t)));
fprintf('Points span x %.2f..%.2f, y %.2f..%.2f\n', ...
    min(P_req(:,1)), max(P_req(:,1)), min(P_req(:,2)), max(P_req(:,2)));
fprintf('Path spans x %.2f..%.2f, y %.2f..%.2f\n', min(P(:,1)), max(P(:,1)), min(P(:,2)), max(P(:,2)));
fprintf('Duration %.1f s; peak %.3f m/s; mean %.3f m/s\n', t(end), max(speed), mean(speed));
assert(max(speed) <= v_limit + 1e-6, 'peak speed above v_limit');

% Write CSV (t, x, y, z), z = 0 because altitude is held by the firmware
csv_path = fullfile(traj_dir, 'elephant_min_snap_traj.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, '# Minimum-snap elephant path from elephant_min_snap_demo.m (x forward, y right, from the start)\n');
fprintf(fid, '# through (cm): %s; fly WITHOUT --traj-rotate\n', strjoin(compose('(%g,%g)', pts_cm), ' '));
fprintf(fid, '# %.1f s, peak %.3f m/s, dt %.3f s\n', t(end), max(speed), dt);
fprintf(fid, 't,x,y,z\n');
fprintf(fid, '%.3f,%.4f,%.4f,0\n', [t P].');
fclose(fid);
fprintf('Wrote %s\n', csv_path);

% Plot: top view (x forward = up) and speed profile
fig = figure('Name', 'Elephant min-snap', 'Color', 'w', 'Position', [100 100 1100 640]);
try, theme(fig, 'light'); catch, end
subplot(1, 2, 1);
plot(P(:,2), P(:,1), 'Color', [0.08 0.45 0.78], 'LineWidth', 2.5); hold on;
plot(P_req(:,2), P_req(:,1), 'ko', 'MarkerFaceColor', [0.95 0.55 0.1], 'MarkerSize', 7);
for k = 1:size(P_req, 1)
    text(P_req(k,2) + 0.07, P_req(k,1) + 0.07, sprintf('%d', k - 1), 'FontWeight', 'bold');
end
step = max(1, round(numel(t) / 35));
quiver(P(1:step:end-1,2), P(1:step:end-1,1), V(1:step:end-1,2), V(1:step:end-1,1), 0.5, ...
       'Color', [0.08 0.45 0.78]);
axis equal; grid on; box on;
xlabel('y right (m)'); ylabel('x forward (m)');
title(sprintf('Min-snap elephant, %.0f s (numbers = point order, 0 = start/land)', t(end)));
subplot(1, 2, 2);
plot(t, speed, 'LineWidth', 1.5); grid on; hold on;
cT = cumsum(T);
xline(cT(pt_idx(2:end-1) - 1), ':', 'Color', [0.6 0.6 0.6]);
xlabel('time (s)'); ylabel('speed (m/s)');
title(sprintf('peak %.3f m/s (dotted = passing a point)', max(speed)));
png_path = fullfile(traj_dir, 'elephant_min_snap.png');
exportgraphics(fig, png_path, 'Resolution', 130);
fprintf('Wrote %s\n', png_path);

function coeff = solveMinSnap(W, T)
% Minimum-snap coefficients: one 7th-order polynomial (8 coefficients) per leg, per axis.
% Minimises the integral of squared snap subject to equality constraints, solved as a KKT system.
M = numel(T);  N = 8 * M;
Q = zeros(N);
for m = 1:M
    idx = 8*(m-1) + (1:8);
    Q(idx, idx) = snapCost() / T(m)^7;
end
coeff = zeros(8, M, 2);
for ax = 1:2
    A = zeros(0, N);  b = zeros(0, 1);
    % Each leg starts at waypoint m and ends at waypoint m+1
    for m = 1:M
        idx = 8*(m-1) + (1:8);
        row = zeros(1, N); row(idx) = basis(0, 0, T(m)); A(end+1, :) = row; b(end+1) = W(m, ax);
        row = zeros(1, N); row(idx) = basis(1, 0, T(m)); A(end+1, :) = row; b(end+1) = W(m+1, ax);
    end
    % Zero velocity, acceleration and jerk at the start and end (rest)
    for k = 1:3
        row = zeros(1, N); row(1:8) = basis(0, k, T(1)); A(end+1, :) = row; b(end+1) = 0;
        row = zeros(1, N); row(8*(M-1) + (1:8)) = basis(1, k, T(M)); A(end+1, :) = row; b(end+1) = 0;
    end
    % Velocity, acceleration and jerk are continuous at every interior waypoint
    for m = 1:M-1
        for k = 1:3
            row = zeros(1, N);
            row(8*(m-1) + (1:8)) = basis(1, k, T(m));
            row(8*m + (1:8))     = -basis(0, k, T(m+1));
            A(end+1, :) = row; b(end+1) = 0;
        end
    end
    % KKT system: minimise c'Qc subject to Ac = b
    K = [2*Q, A.'; A, zeros(size(A, 1))];
    sol = K \ [zeros(N, 1); b(:)];
    coeff(:, :, ax) = reshape(sol(1:N), 8, M);
end
end

function pk = legPeaks(coeff, T)
% Peak speed on each leg
pk = zeros(1, numel(T));
for m = 1:numel(T)
    tau = linspace(0, 1, 120);
    v = zeros(numel(tau), 2);
    for ax = 1:2
        for n = 1:numel(tau)
            v(n, ax) = basis(tau(n), 1, T(m)) * coeff(:, m, ax);
        end
    end
    pk(m) = max(vecnorm(v, 2, 2));
end
end

function Qn = snapCost()
% Integral over tau in [0, 1] of the squared 4th derivative, for powers 0..7
Qn = zeros(8);
for i = 4:7
    for j = 4:7
        Qn(i+1, j+1) = (factorial(i)/factorial(i-4)) * (factorial(j)/factorial(j-4)) / (i + j - 7);
    end
end
end

function row = basis(tau, k, Tm)
% k-th time derivative of [1 tau ... tau^7] at tau, in physical time
row = zeros(1, 8);
for j = k:7
    row(j+1) = factorial(j) / factorial(j-k) * tau^(j-k);
end
row = row / Tm^k;
end

function [P, V] = sampleMinSnap(coeff, T, t)
% Position and velocity of the polynomials at times t
edges = [0 cumsum(T)];
P = zeros(numel(t), 2);  V = zeros(numel(t), 2);
for n = 1:numel(t)
    m = min(find(t(n) >= edges, 1, 'last'), numel(T));
    tau = (t(n) - edges(m)) / T(m);
    for ax = 1:2
        c = coeff(:, m, ax);
        P(n, ax) = basis(tau, 0, T(m)) * c;
        V(n, ax) = basis(tau, 1, T(m)) * c;
    end
end
end
