clc; close all;
traj_dir = fileparts(mfilename('fullpath'));

use_sparse = true;
pts_full = [  0    0;   140  -90;   200  -90;   320    0;   380  100;   250  120;   140  100;
             80  -20;   140 -140;   250 -160;   380 -140;   320    0;   200   65;   140   65;
              0    0];
pts_sparse = [    0    0;
               140  -90;
               380  100;
               140  100;
                80  -20;
               140 -140;
               380 -140;
               140   65;
                 0    0];
if use_sparse, pts_cm = pts_sparse; else, pts_cm = pts_full; end

v_limit   = 0.297;
accel     = 0.08;
lateral   = 0.04;
dt        = 0.05;

P_req = pts_cm / 100;

path = catmullRom(P_req, 60);

[t, P, V, speed] = timePath(path, v_limit, accel, lateral, dt);

fprintf('%d points, %.2f m of path\n', size(P_req, 1), sum(vecnorm(diff(P), 2, 2)));
fprintf('Points span x %.2f..%.2f, y %.2f..%.2f\n', min(P_req(:,1)), max(P_req(:,1)), min(P_req(:,2)), max(P_req(:,2)));
fprintf('Path spans x %.2f..%.2f, y %.2f..%.2f\n', min(P(:,1)), max(P(:,1)), min(P(:,2)), max(P(:,2)));
fprintf('Duration %.1f s; peak %.3f m/s; mean %.3f m/s\n', t(end), max(speed), mean(speed));
assert(max(speed) <= v_limit + 0.005, 'peak speed above v_limit');
overshoot = max([min(P_req(:,1)) - min(P(:,1)), max(P(:,1)) - max(P_req(:,1)), ...
                 min(P_req(:,2)) - min(P(:,2)), max(P(:,2)) - max(P_req(:,2)), 0]);
fprintf('Overshoot beyond the points: %.2f m\n', overshoot);

csv_path = fullfile(traj_dir, 'elephant_catmull_rom_traj.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, '# Catmull-Rom elephant path from elephant_catmull_rom_demo.m (x forward, y right, from the start)\n');
fprintf(fid, '# through (cm): %s; fly WITHOUT --traj-rotate\n', strjoin(compose('(%g,%g)', pts_cm), ' '));
fprintf(fid, '# %.1f s, peak %.3f m/s, dt %.3f s\n', t(end), max(speed), dt);
fprintf(fid, 't,x,y,z\n');
fprintf(fid, '%.3f,%.4f,%.4f,0\n', [t P].');
fclose(fid);
fprintf('Wrote %s\n', csv_path);

fig = figure('Name', 'Elephant Catmull-Rom', 'Color', 'w', 'Position', [100 100 1100 640]);
try, theme(fig, 'light'); catch, end
subplot(1, 2, 1);
plot(P(:,2), P(:,1), 'Color', [0.85 0.33 0.1], 'LineWidth', 2.5); hold on;
plot(P_req(:,2), P_req(:,1), 'ko', 'MarkerFaceColor', [0.95 0.55 0.1], 'MarkerSize', 7);
for k = 1:size(P_req, 1)
    text(P_req(k,2) + 0.07, P_req(k,1) + 0.07, sprintf('%d', k - 1), 'FontWeight', 'bold');
end
step = max(1, round(numel(t) / 35));
quiver(P(1:step:end-1,2), P(1:step:end-1,1), V(1:step:end-1,2), V(1:step:end-1,1), 0.5, ...
       'Color', [0.85 0.33 0.1]);
axis equal; grid on; box on;
xlabel('y right (m)'); ylabel('x forward (m)');
title(sprintf('Catmull-Rom elephant (%d pts), %.0f s, overshoot %.2f m', size(P_req,1), t(end), overshoot));
subplot(1, 2, 2);
plot(t, speed, 'LineWidth', 1.5); grid on;
xlabel('time (s)'); ylabel('speed (m/s)');
title(sprintf('peak %.3f m/s', max(speed)));
png_path = fullfile(traj_dir, 'elephant_catmull_rom.png');
exportgraphics(fig, png_path, 'Resolution', 130);
fprintf('Wrote %s\n', png_path);

function out = catmullRom(pts, per_seg)

P = [pts(1,:); pts; pts(end,:)];
out = zeros(0, 2);
tj = @(ti, a, b) ti + max(norm(b - a), 1e-6)^0.5;
for i = 2:size(P, 1) - 2
    p0 = P(i-1,:);  p1 = P(i,:);  p2 = P(i+1,:);  p3 = P(i+2,:);
    t0 = 0;  t1 = tj(t0, p0, p1);  t2 = tj(t1, p1, p2);  t3 = tj(t2, p2, p3);
    tt = linspace(t1, t2, per_seg + 1);  tt(end) = [];
    for t = tt
        a1 = (t1-t)/(t1-t0)*p0 + (t-t0)/(t1-t0)*p1;
        a2 = (t2-t)/(t2-t1)*p1 + (t-t1)/(t2-t1)*p2;
        a3 = (t3-t)/(t3-t2)*p2 + (t-t2)/(t3-t2)*p3;
        b1 = (t2-t)/(t2-t0)*a1 + (t-t0)/(t2-t0)*a2;
        b2 = (t3-t)/(t3-t1)*a2 + (t-t1)/(t3-t1)*a3;
        out(end+1, :) = (t2-t)/(t2-t1)*b1 + (t-t1)/(t2-t1)*b2;
    end
end
out(end+1, :) = P(end-1, :);
end

function [t, P, V, speed] = timePath(path, v_cruise, a_max, a_lat, dt)

ds_raw = vecnorm(diff(path), 2, 2);
s_raw = [0; cumsum(ds_raw)];
L = s_raw(end);
s = [(0:0.005:L - 1e-9)'; L];
Q = [interp1(s_raw, path(:,1), s), interp1(s_raw, path(:,2), s)];
ds = diff(s);

heading = unwrap(atan2(diff(Q(:,2)), diff(Q(:,1))));
kappa = abs(gradient(heading, s(1:end-1)));
k = 21;
kp = conv([kappa(1)*ones(k,1); kappa; kappa(end)*ones(k,1)], ones(k,1)/k, 'same');
kappa = [kp(k+1:end-k); kp(end-k)];

v_lim = min(v_cruise, sqrt(a_lat ./ max(kappa, 1e-6)));
v_lim(1) = 0;  v_lim(end) = 0;
v2 = v_lim.^2;
for i = 2:numel(v2),        v2(i) = min(v2(i), v2(i-1) + 2*a_max*ds(i-1)); end
for i = numel(v2)-1:-1:1,   v2(i) = min(v2(i), v2(i+1) + 2*a_max*ds(i));   end
v = sqrt(v2);
tt = [0; cumsum(ds ./ max((v(1:end-1) + v(2:end))/2, 1e-4))];

t  = (0:dt:tt(end))';
if t(end) < tt(end), t(end+1) = tt(end); end
sk = interp1(tt, s, t, 'linear', L);
P  = [interp1(s, Q(:,1), sk), interp1(s, Q(:,2), sk)];
V  = [gradient(P(:,1), t), gradient(P(:,2), t)];
speed = vecnorm(V, 2, 2);
end
