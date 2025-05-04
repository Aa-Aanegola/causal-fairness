load_resp_data <- function(src) {

  dat <- load_concepts(
    c("vent_ind", "resp", "po2", "sofa", "o2sat", "sex", "age"), src,
    verbose = TRUE
  )

  dat <- dat[get(index_var(dat)) <= hours(48L) &
               get(index_var(dat)) >= hours(0L)]
  dat[is.na(vent_ind), vent_ind := FALSE]

  dat[, is_vent := cummax(vent_ind), by = c(id_vars(dat))]

  cand <- unique(id_col(dat[o2sat <= 96 & is_vent == 0]))
  cdat <- dat[id_col(dat) %in% cand]
  cdat <- replace_na(cdat, type = "locf", vars = c("o2sat", "po2", "resp"))
  cdat[is.na(po2), po2 := median(po2, na.rm = TRUE)]
  cdat[is.na(resp), resp := median(resp, na.rm = TRUE)]

  # lag by 3 hours both ways
  cdat[, is_vent_lag3 := data.table::shift(is_vent, -3L)]
  cdat[, is_vent_lagrev3 := data.table::shift(is_vent, 3L)]

  # the actioned cohort
  act <- merge(
    cdat[is_vent == 0 & is_vent_lag3 == 1,
         list(o2prior = mean(o2sat, na.rm = TRUE), sofa = max(sofa),
              resp = mean(resp, na.rm = TRUE), po2 = mean(po2, na.rm = TRUE),
              sex = unique(sex), age = unique(age)),
         by = c(id_vars(dat))],
    cdat[is_vent == 1 & is_vent_lagrev3 == 0,
         list(o2post = mean(o2sat, na.rm = TRUE)),
         by = c(id_vars(dat))]
  )

  act[, respirator := 1]

  # take complete cases
  act <- act[complete.cases(act)]

  # the non-actioned cohort
  ctrls <- id_col(cdat[, max(is_vent), by = c(id_vars(cdat))][V1 == 0])
  ndat <- cdat[(id_col(cdat) %in% ctrls)]

  skp <- merge(
    ndat[get(index_var(ndat)) %in% hours(10, 11, 12),
         list(o2prior = mean(o2sat, na.rm = TRUE), sofa = max(sofa),
              resp = mean(resp, na.rm = TRUE), po2 = mean(po2, na.rm = TRUE),
              sex = unique(sex), age = unique(age)),
         by = c(id_vars(dat))],
    ndat[get(index_var(ndat)) %in% hours(13, 14, 15),
         list(o2post = mean(o2sat, na.rm = TRUE)),
         by = c(id_vars(dat))]
  )
  skp <- skp[, respirator  := 0]
  skp <- skp[complete.cases(skp)]

  res <- rbind(act, skp)
  res[, sex := ifelse(sex == "Male", 1, 0)]
  res
}

# load_resp_data_with_chexpert <- function(src) {
#   cxr_findings <- c("atelectasis", "cardiomegaly", "consolidation", "edema",
#                   "enlarged_cardiomediastinum", "fracture", "lung_lesion",
#                   "lung_opacity", "no_finding", "pleural_effusion",
#                   "pleural_other", "pneumonia", "pneumothorax", "support_devices")

#   dat <- load_concepts(
#     c("vent_ind", "resp", "po2", "sofa", "o2sat", "sex", "age"), src,
#     verbose = TRUE
#   )

#   cxr <- load_concepts(
#     c(cxr_findings, "dicom_id"), src, verbose = TRUE
#   )

#   cxr <- cxr[get(index_var(cxr)) <= hours(0L) & 
#                get(index_var(cxr)) >= hours(-8L)]

#   cxr <- cxr[, lapply(.SD, function(x) {
#           if (any(x == 1, na.rm = TRUE)) return(1L)
#           else if (any(x == -1, na.rm = TRUE)) return(-1L)
#           else return(0L)
#      }), by = .(stay_id, dicom_id), .SDcols = cxr_findings]

#   dat <- dat[get(index_var(dat)) <= hours(48L) &
#                get(index_var(dat)) >= hours(0L)]
#   dat[is.na(vent_ind), vent_ind := FALSE]

#   dat[, is_vent := cummax(vent_ind), by = c(id_vars(dat))]

#   cand <- unique(id_col(dat[o2sat <= 96 & is_vent == 0]))
#   cdat <- dat[id_col(dat) %in% cand]
#   cdat <- replace_na(cdat, type = "locf", vars = c("o2sat", "po2", "resp"))
#   cdat[is.na(po2), po2 := median(po2, na.rm = TRUE)]
#   cdat[is.na(resp), resp := median(resp, na.rm = TRUE)]

#   # lag by 3 hours both ways
#   cdat[, is_vent_lag3 := data.table::shift(is_vent, -3L)]
#   cdat[, is_vent_lagrev3 := data.table::shift(is_vent, 3L)]

#   # the actioned cohort
#   act <- merge(
#     cdat[is_vent == 0 & is_vent_lag3 == 1,
#          list(o2prior = mean(o2sat, na.rm = TRUE), sofa = max(sofa),
#               resp = mean(resp, na.rm = TRUE), po2 = mean(po2, na.rm = TRUE),
#               sex = unique(sex), age = unique(age)),
#          by = c(id_vars(dat))],
#     cdat[is_vent == 1 & is_vent_lagrev3 == 0,
#          list(o2post = mean(o2sat, na.rm = TRUE)),
#          by = c(id_vars(dat))]
#   )

#   act[, respirator := 1]

#   # take complete cases
#   act <- act[complete.cases(act)]

#   act <- merge(act, cxr, by = c(id_vars(dat)), all.x = TRUE)
#   act[, (cxr_findings) := lapply(.SD, function(x) ifelse(is.na(x), -2L, x)), .SDcols = cxr_findings]

#   # the non-actioned cohort
#   ctrls <- id_col(cdat[, max(is_vent), by = c(id_vars(cdat))][V1 == 0])
#   ndat <- cdat[(id_col(cdat) %in% ctrls)]

#   skp <- merge(
#     ndat[get(index_var(ndat)) %in% hours(10, 11, 12),
#          list(o2prior = mean(o2sat, na.rm = TRUE), sofa = max(sofa),
#               resp = mean(resp, na.rm = TRUE), po2 = mean(po2, na.rm = TRUE),
#               sex = unique(sex), age = unique(age)),
#          by = c(id_vars(dat))],
#     ndat[get(index_var(ndat)) %in% hours(13, 14, 15),
#          list(o2post = mean(o2sat, na.rm = TRUE)),
#          by = c(id_vars(dat))]
#   )
#   skp <- skp[, respirator  := 0]
#   skp <- skp[complete.cases(skp)]

#   skp <- merge(skp, cxr, by = c(id_vars(dat)), all.x = TRUE)
#   skp[, (cxr_findings) := lapply(.SD, function(x) ifelse(is.na(x), -2L, x)), .SDcols = cxr_findings]

#   res <- rbind(act, skp)
#   res[, sex := ifelse(sex == "Male", 1, 0)]
#   res
# }


load_resp_data_with_chexpert <- function(src) {
  cxr_findings <- c("atelectasis", "cardiomegaly", "consolidation", "edema",
                    "enlarged_cardiomediastinum", "fracture", "lung_lesion",
                    "lung_opacity", "no_finding", "pleural_effusion",
                    "pleural_other", "pneumonia", "pneumothorax", "support_devices")

  # Load EHR and CXR data
  dat <- load_concepts(
    c("vent_ind", "resp", "po2", "sofa", "o2sat", "sex", "age"), src, verbose = TRUE
  )
  cxr <- load_concepts(cxr_findings, src, verbose = TRUE)

  # Trim to 48h post admission for vitals, fill missing
  dat <- dat[get(index_var(dat)) <= hours(48L) & get(index_var(dat)) >= hours(0L)]
  dat[is.na(vent_ind), vent_ind := FALSE]
  dat[, is_vent := cummax(vent_ind), by = c(id_vars(dat))]

  # Candidate patients (hypoxemic but not yet intubated)
  cand <- unique(id_col(dat[o2sat <= 96 & is_vent == 0]))
  cdat <- dat[id_col(dat) %in% cand]
  cdat <- replace_na(cdat, type = "locf", vars = c("o2sat", "po2", "resp"))
  cdat[is.na(po2), po2 := median(po2, na.rm = TRUE)]
  cdat[is.na(resp), resp := median(resp, na.rm = TRUE)]

  # Lags for detecting transitions
  cdat[, is_vent_lag3 := data.table::shift(is_vent, -3L)]
  cdat[, is_vent_lagrev3 := data.table::shift(is_vent, 3L)]

  # Get time of first ventilation
  intub_time <- dat[is_vent == 1, .(intub_time = min(get(index_var(dat)))), by = stay_id]

  # Filter CXR to 12h before intubation, keep most recent one
  cxr_aligned <- merge(cxr, intub_time, by = "stay_id")
  cxr_aligned <- cxr_aligned[
    get(index_var(cxr)) <= intub_time & 
    get(index_var(cxr)) >= intub_time - hours(12)
  ]
  cxr_aligned <- cxr_aligned[
  get(index_var(cxr)) >= intub_time - hours(24) & 
  get(index_var(cxr)) <= intub_time
]
cxr_aligned <- cxr_aligned[order(get(index_var(cxr))), .SD[.N], by = stay_id]

  # Collapse to one row per stay_id, keep only findings
  cxr_final <- cxr_aligned[, lapply(.SD, function(x) {
    if (any(x == 1, na.rm = TRUE)) return(1L)
    else if (any(x == -1, na.rm = TRUE)) return(-1L)
    else return(0L)
  }), by = stay_id, .SDcols = cxr_findings]

  # Actioned cohort
  act <- merge(
    cdat[is_vent == 0 & is_vent_lag3 == 1,
         .(o2prior = mean(o2sat, na.rm = TRUE), sofa = max(sofa),
           resp = mean(resp, na.rm = TRUE), po2 = mean(po2, na.rm = TRUE),
           sex = unique(sex), age = unique(age)),
         by = c(id_vars(dat))],
    cdat[is_vent == 1 & is_vent_lagrev3 == 0,
         .(o2post = mean(o2sat, na.rm = TRUE)),
         by = c(id_vars(dat))]
  )
  act[, respirator := 1L]
  act <- act[complete.cases(act)]
  act <- merge(act, cxr_final, by = "stay_id", all.x = TRUE)

  # Control cohort — estimate median anchor time
  ctrls <- id_col(cdat[, max(is_vent), by = c(id_vars(cdat))][V1 == 0])
  ndat <- cdat[id_col(cdat) %in% ctrls]

  # Approximate matched anchor (e.g., 24h) for controls
  ctrl_anchor <- ndat[, .(anchor = median(get(index_var(ndat)))), by = stay_id]
  cxr_ctrl <- merge(cxr, ctrl_anchor, by = "stay_id")
  cxr_ctrl <- cxr_ctrl[
    get(index_var(cxr)) <= anchor & 
    get(index_var(cxr)) >= anchor - hours(12)
  ]
  cxr_ctrl <- cxr_ctrl[
  get(index_var(cxr)) >= anchor - hours(24) & 
  get(index_var(cxr)) <= anchor 
  ]
  cxr_ctrl <- cxr_ctrl[order(get(index_var(cxr))), .SD[.N], by = stay_id]
  
  cxr_ctrl_final <- cxr_ctrl[, lapply(.SD, function(x) {
    if (any(x == 1, na.rm = TRUE)) return(1L)
    else if (any(x == -1, na.rm = TRUE)) return(-1L)
    else return(0L)
  }), by = stay_id, .SDcols = cxr_findings]

  # Control features
  skp <- merge(
    ndat[get(index_var(ndat)) %in% hours(10, 11, 12),
         .(o2prior = mean(o2sat, na.rm = TRUE), sofa = max(sofa),
           resp = mean(resp, na.rm = TRUE), po2 = mean(po2, na.rm = TRUE),
           sex = unique(sex), age = unique(age)),
         by = c(id_vars(dat))],
    ndat[get(index_var(ndat)) %in% hours(13, 14, 15),
         .(o2post = mean(o2sat, na.rm = TRUE)),
         by = c(id_vars(dat))]
  )
  skp[, respirator := 0L]
  skp <- skp[complete.cases(skp)]
  skp <- merge(skp, cxr_ctrl_final, by = "stay_id", all.x = TRUE)

  # Combine cohorts
  res <- rbind(act, skp, fill = TRUE)
  res[, sex := fifelse(sex == "Male", 1L, 0L)]
  return(res)
}


load_cxr_with_outcomes <- function(src, hours_window = 6L) {
  # CheXpert labels (you may also join embeddings later by study_id)
  cxr_findings <- c("atelectasis", "cardiomegaly", "consolidation", "edema",
                    "enlarged_cardiomediastinum", "fracture", "lung_lesion",
                    "lung_opacity", "no_finding", "pleural_effusion",
                    "pleural_other", "pneumonia", "pneumothorax", "support_devices")

  # Load EHR concepts
  ehr <- load_concepts(
    c("o2sat", "resp", "po2", "sofa", "age", "sex", "hr", "sbp", "dbp"), 
    src, verbose = TRUE
  )
  ehr <- ehr[get(index_var(ehr)) >= hours(-hours_window) & get(index_var(ehr)) <= hours(hours_window)]
  ehr[, sex := ifelse(sex == "Male", 1L, 
             ifelse(sex == "Female", 0L, NA_integer_))]
  group_cols <- id_vars(ehr)
  ehr <- ehr[, lapply(.SD, mean, na.rm = TRUE), 
           by = group_cols, 
           .SDcols = setdiff(names(ehr), c(group_cols, index_var(ehr)))]


  # Load CheXpert labels (X-ray view)
  cxr <- load_concepts(cxr_findings, src, verbose = TRUE)
  cxr <- cxr[, lapply(.SD, function(x) {
    if (any(x == 1, na.rm = TRUE)) return(1L)
    else if (any(x == -1, na.rm = TRUE)) return(-1L)
    else return(0L)
  }), by = .(stay_id), .SDcols = cxr_findings]

  # Load outcomes
  meds <- load_concepts("lasix", src, verbose = TRUE)  # furosemide
  lasix_admin <- meds[, .(lasix_given = any(!is.na(val))), by = id_vars(meds)]

  adm <- load_concepts("admittime", src, verbose = TRUE)
  mort <- load_concepts("deathtime", src, verbose = TRUE)
  adm_mort <- merge(adm, mort, by = "subject_id", all.x = TRUE)
  adm_mort[, mortality := !is.na(deathtime)]

  # Length of stay
  adm_mort[, los_days := as.numeric(difftime(dischtime, admittime, units = "days"))]

  # Merge everything by patient stay
  out <- Reduce(function(x, y) merge(x, y, by = "stay_id", all = TRUE),
                list(ehr, cxr, lasix_admin, adm_mort))

  out[, sex := ifelse(sex == "Male", 1L, 0L)]

  return(out)
}