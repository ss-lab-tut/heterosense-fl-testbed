"""HeteroSense-FL quick demo — entry point: heterosense-demo"""
import numpy as np
from heterosense import ClientFactory, ConfigurationManager as CM
from heterosense import DatasetBuilder, TemporalWindowSampler, run_validation


def main():
    print("=== HeteroSense-FL Quick Demo ===\n")
    clients = ClientFactory.make(3, strategy="round_robin")
    cfg = CM.from_clients(clients, n_steps=500)
    sc = cfg.to_sim_config()
    data = DatasetBuilder(sc).build()

    print("Dataset:")
    for cid, bundles in data.items():
        cc = next(c for c in sc.clients if c.client_id == cid)
        has_l = any(b.lidar    is not None for b in bundles)
        has_p = any(b.pressure is not None for b in bundles)
        print(f"  client {cid}: modalities={list(cc.channel_availability)}"
              f"  lidar={has_l}  bed={has_p}")

    print("\nTemporalWindowSampler (window=3):")
    sampler = TemporalWindowSampler(data["0"], window=3)
    for i, w in enumerate(sampler):
        if i >= 3: break
        z = TemporalWindowSampler.lidar_z_series(w)
        p = TemporalWindowSampler.pressure_series(w)
        lbl = TemporalWindowSampler.center_label(w, sampler.center_idx())
        print(f"  window {i}: label={lbl}  z_mean={z.mean():.3f}  p_mean={p.mean():.3f}")

    print("\nValidation (V1-V4):")
    cm = {c.client_id: list(c.channel_availability) for c in sc.clients}
    for r in run_validation(data, cm):
        icon = "OK" if r.passed else "FAIL"
        print(f"  [{icon}] {r.name}: {r.reason}")

    print("\nDone. Run benchmark: heterosense-benchmark")


if __name__ == "__main__":
    main()
