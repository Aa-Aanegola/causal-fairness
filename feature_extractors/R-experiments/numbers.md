# Statistics that are important

- For the cxr data, there's 18355 patients who received a chest x-ray within 36 hours before being admitted and 24 hours after being admitted. `unique(cxr[studytime >= hours(-36L) & studytime <= hours(24L), "stay_id"])`

```R
dat <- load_concepts(
    c("vent_ind", "resp", "po2", "sofa", "o2sat", "sex", "age"), src,
    verbose = TRUE
  )
cxr <- load_concepts(cxr_findings, src)

d2 <- dat[get(index_var(dat)) <= hours(48L) &
               get(index_var(dat)) >= hours(0L)]
unique(d2[, "stay_id"])
# Output = 73,173

length(intersect(d2[["stay_id"]], cxr[["stay_id"]]))
# Output 26509

d2 <- d2[is.na(vent_ind), vent_ind := FALSE]


length(intersect(cand, cxr[["stay_id"]]))
[1] 18290

> length(intersect(unique(act[["stay_id"]]), unique(cxr[["stay_id"]])))
[1] 1734
> length(act[["stay_id"]])
[1] 5534
> length(skp[["stay_id"]])
[1] 43706
> length(intersect(unique(skp[["stay_id"]]), unique(cxr[["stay_id"]])))
[1] 10485

cxr_collapsed <- cxr_sub[, lapply(.SD, function(x) {
    if (any(x == 1, na.rm = TRUE)) return(1L)
    else if (any(x == -1, na.rm = TRUE)) return(-1L)
    else return(0L)
  }), by = stay_id, .SDcols = cxr_findings]

> act_w_cxr[is.na(atelectasis), .N]
[1] 3800
> act_w_cxr[!is.na(atelectasis), .N]
[1] 1734

> skp_w_cxr[!is.na(atelectasis), .N]
[1] 10485
> skp_w_cxr[is.na(atelectasis), .N]
[1] 33221


```