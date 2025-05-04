library(data.table)
library(ricu)
src <- "miiv"
mdat <- load_resp_data(src)
xray <- fread("/local/eb/aa5506/feature_extractors/torchxray/torchxray-resnet-features-metadata.csv")
pxr <- function(tab) tab[, .(subject_id, study_id, ViewPosition, StudyDate, emb_9)]
xray[, datetime := as.POSIXct(paste0(StudyDate, sprintf("%06.0f", as.numeric(StudyTime))), format = "%Y%m%d%H%M%S")]
temp <- merge(xray, miiv$icustays, by = c("subject_id"), all=TRUE)
temp <- temp[datetime >= intime & datetime <= outtime]
resp_w_emb <- merge(mdat, temp, by = c("stay_id"))