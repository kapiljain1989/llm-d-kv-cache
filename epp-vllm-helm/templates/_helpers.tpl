{{/* vim: set filetype=gotpl: */}}
{{/*
Expand the name of the chart.
*/}}
{{- define "epp-vllm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "epp-vllm.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart label
*/}}
{{- define "epp-vllm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "epp-vllm.labels" -}}
helm.sh/chart: {{ include "epp-vllm.chart" . }}
{{ include "epp-vllm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "epp-vllm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "epp-vllm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
vLLM image
*/}}
{{- define "epp-vllm.vllmImage" -}}
{{- $tag := default .Chart.AppVersion .Values.vllm.image.tag -}}
{{- printf "%s:%s" .Values.vllm.image.repository $tag -}}
{{- end -}}

{{/*
EPP / inference-scheduler image
*/}}
{{- define "epp-vllm.eppImage" -}}
{{- printf "%s/%s:%s" .Values.inferenceExtension.image.hub .Values.inferenceExtension.image.name .Values.inferenceExtension.image.tag -}}
{{- end -}}

{{/*
EPP service name
*/}}
{{- define "epp-vllm.eppServiceName" -}}
{{- printf "%s-inference-scheduler" .Release.Name -}}
{{- end -}}

{{/*
EPP ZMQ endpoint URL (for vLLM to connect to)
*/}}
{{- define "epp-vllm.eppZmqUrl" -}}
{{- $svcName := include "epp-vllm.eppServiceName" . -}}
{{- printf "tcp://%s.%s.svc.cluster.local:5557" $svcName .Release.Namespace -}}
{{- end -}}

{{/*
Secret key name
*/}}
{{- define "epp-vllm.secretKeyName" -}}
{{- printf "%s_%s" .Values.secret.keyPrefix .Values.vllm.model.label -}}
{{- end -}}

{{/*
PVC name
*/}}
{{- define "epp-vllm.pvcName" -}}
{{- printf "%s-%s-storage" (include "epp-vllm.fullname" .) .Values.vllm.model.label | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
InferencePool name
*/}}
{{- define "epp-vllm.poolName" -}}
{{- printf "%s-pool" .Release.Name -}}
{{- end -}}

{{/*
Renders imagePullSecrets block
*/}}
{{- define "epp-vllm.imagePullSecrets" -}}
{{- $secrets := .componentSecrets | default .globalSecrets -}}
{{- if $secrets -}}
imagePullSecrets:
{{- toYaml $secrets | nindent 2 }}
{{- end -}}
{{- end -}}
