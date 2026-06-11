#!/bin/bash

# Shared schema-backed engine helpers.
#
# Engines keep their own schema.sh files and call `schema_engine_init` with:
#   schema_engine_init "qdrant" "QDRANT" "Qdrant"
#
# This defines the engine-specific register function expected by schema.sh,
# e.g. register_qdrant_var, while the actual registry behavior lives here.

register_common_var() {
    if [[ -z "${SCHEMA_ACTIVE_PREFIX:-}" ]]; then
        echo "register_common_var called without an active schema prefix" >&2
        return 1
    fi

    schema_register_var "$SCHEMA_ACTIVE_PREFIX" "$@"
}

# Initialize one schema registry and create the engine-specific registration
# function that schema.sh will call, e.g. register_qdrant_var.
schema_engine_init() {
    local schema_lower="$1"
    local schema_prefix="$2"
    local display_name="$3"

    printf -v "${schema_prefix}_DISPLAY_NAME" '%s' "$display_name"
    declare -ag "${schema_prefix}_VAR_ORDER=()"
    declare -gA "${schema_prefix}_REQUIRED_KIND=()"
    declare -gA "${schema_prefix}_REQUIRED_IF=()"
    declare -gA "${schema_prefix}_DEFAULT=()"
    declare -gA "${schema_prefix}_CHOICES=()"
    declare -gA "${schema_prefix}_DESC=()"
    declare -gA "${schema_prefix}_VALUES=()"

    eval "register_${schema_lower}_var() { schema_register_var \"$schema_prefix\" \"\$@\"; }"
}

# Clear a schema registry before re-sourcing schema.sh.
schema_reset() {
    local schema_prefix="$1"
    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local -n required_kind_ref="${schema_prefix}_REQUIRED_KIND"
    local -n required_if_ref="${schema_prefix}_REQUIRED_IF"
    local -n default_ref="${schema_prefix}_DEFAULT"
    local -n choices_ref="${schema_prefix}_CHOICES"
    local -n desc_ref="${schema_prefix}_DESC"
    local -n values_ref="${schema_prefix}_VALUES"

    order_ref=()
    required_kind_ref=()
    required_if_ref=()
    default_ref=()
    choices_ref=()
    desc_ref=()
    values_ref=()
}

# Add one variable definition to the active registry.
#
# Arguments after schema_prefix are:
#   name required_kind default choices description [required_if]
# required_kind is usually required, optional, or conditional.
schema_register_var() {
    local schema_prefix="$1"
    local name="$2"
    local required_kind="$3"
    local default_value="$4"
    local choices="$5"
    local description="$6"
    local required_if="${7:-}"

    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local -n required_kind_ref="${schema_prefix}_REQUIRED_KIND"
    local -n required_if_ref="${schema_prefix}_REQUIRED_IF"
    local -n default_ref="${schema_prefix}_DEFAULT"
    local -n choices_ref="${schema_prefix}_CHOICES"
    local -n desc_ref="${schema_prefix}_DESC"

    order_ref+=("$name")
    required_kind_ref["$name"]="$required_kind"
    required_if_ref["$name"]="$required_if"
    default_ref["$name"]="$default_value"
    choices_ref["$name"]="$choices"
    desc_ref["$name"]="$description"
}

# Reload schema.sh into a clean registry.
schema_load() {
    local schema_prefix="$1"
    local schema_file="$2"

    schema_reset "$schema_prefix"
    SCHEMA_ACTIVE_PREFIX="$schema_prefix" source "$schema_file"
}

# Append another schema file into an existing registry without clearing it.
schema_append_file() {
    local schema_prefix="$1"
    local schema_file="$2"

    SCHEMA_ACTIVE_PREFIX="$schema_prefix" source "$schema_file"
}

# Split a raw schema value into sweep entries. Empty values are preserved as a
# single empty entry so optional blank variables still participate in combos.
schema_split_raw_values() {
    local raw_value="$1"
    local output_name="$2"
    local -n output_ref="$output_name"

    if [[ -z "$raw_value" ]]; then
        output_ref=("")
    else
        read -r -a output_ref <<< "$raw_value"
        if (( ${#output_ref[@]} == 0 )); then
            output_ref=("")
        fi
    fi
}

# Return the scalar value used for global variables before combo expansion.
schema_first_raw_value() {
    local raw_value="$1"
    local values=()

    schema_split_raw_values "$raw_value" values
    printf '%s\n' "${values[0]}"
}

# Populate shell globals from the first value of each registered variable.
#
# This lets validation and summary code use normal variable names even though
# the canonical source is the registry's VALUES map.
schema_assign_globals_from_values() {
    local schema_prefix="$1"
    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local -n values_ref="${schema_prefix}_VALUES"
    local var_name
    local first_value

    for var_name in "${order_ref[@]}"; do
        first_value="$(schema_first_raw_value "${values_ref[$var_name]-}")"
        printf -v "$var_name" '%s' "$first_value"
    done

    if [[ -n "${QUEUE:-}" ]]; then
        queue="$QUEUE"
    fi
}

# Pull current shell globals back into the registry.
#
# This keeps legacy override paths and any engine-specific default mutations in
# sync before environment overrides are applied.
schema_sync_values_from_current_globals() {
    local schema_prefix="$1"
    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local -n values_ref="${schema_prefix}_VALUES"
    local var_name
    local raw_value

    for var_name in "${order_ref[@]}"; do
        if raw_value="$(get_var_as_raw_string "$var_name" 2>/dev/null)"; then
            values_ref["$var_name"]="$raw_value"
        fi
    done
}

# Seed the runtime VALUES map from schema defaults.
schema_apply_defaults() {
    local schema_prefix="$1"
    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local -n values_ref="${schema_prefix}_VALUES"
    local -n default_ref="${schema_prefix}_DEFAULT"
    local var_name

    for var_name in "${order_ref[@]}"; do
        values_ref["$var_name"]="${default_ref[$var_name]}"
    done
}

# Apply NAME_OVERRIDE or UPPERCASE_NAME_OVERRIDE values into the registry.
#
# The root submit manager translates --set NAME=value into these override
# variables before loading the engine.
schema_apply_overrides_from_env() {
    local schema_prefix="$1"
    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local -n values_ref="${schema_prefix}_VALUES"
    local var_name
    local override_name
    local uppercase_override_name

    for var_name in "${order_ref[@]}"; do
        override_name="${var_name}_OVERRIDE"
        uppercase_override_name="${var_name^^}_OVERRIDE"

        if [[ -n "${!override_name:-}" ]]; then
            values_ref["$var_name"]="${!override_name}"
        elif [[ -n "${!uppercase_override_name:-}" ]]; then
            values_ref["$var_name"]="${!uppercase_override_name}"
        fi
    done
}

# Return success when value is allowed by the optional choices list.
schema_value_in_choices() {
    local value="$1"
    local choices_raw="$2"
    local choices=()
    local choice

    [[ -z "$choices_raw" ]] && return 0
    [[ -z "$value" ]] && return 0

    schema_split_raw_values "$choices_raw" choices
    for choice in "${choices[@]}"; do
        if [[ "$value" == "$choice" ]]; then
            return 0
        fi
    done

    return 1
}

# Evaluate a simple conditional requirement expression.
#
# Supported format: VAR=value or VAR=value1|value2. The current scalar value of
# VAR is compared against the allowed values.
schema_condition_matches_current() {
    local condition="$1"
    local lhs
    local rhs
    local current
    local allowed_values=()
    local allowed

    [[ -z "$condition" ]] && return 0

    lhs="${condition%%=*}"
    rhs="${condition#*=}"
    current="$(get_var_as_raw_string "$lhs" 2>/dev/null || true)"
    current="$(schema_first_raw_value "$current")"

    schema_split_raw_values "${rhs//|/ }" allowed_values
    for allowed in "${allowed_values[@]}"; do
        if [[ "$current" == "$allowed" ]]; then
            return 0
        fi
    done

    return 1
}

# Return success when a variable is required under the current settings.
schema_var_is_required_current() {
    local schema_prefix="$1"
    local var_name="$2"
    local -n required_kind_ref="${schema_prefix}_REQUIRED_KIND"
    local -n required_if_ref="${schema_prefix}_REQUIRED_IF"

    case "${required_kind_ref[$var_name]}" in
        required)
            return 0
            ;;
        conditional)
            schema_condition_matches_current "${required_if_ref[$var_name]}"
            return $?
            ;;
        *)
            return 1
            ;;
    esac
}

# Validate all current values for required-ness and choices.
#
# Sweep variables validate every emitted value, not just the first scalar.
schema_validate_current_values() {
    local schema_prefix="$1"
    local -n display_name_ref="${schema_prefix}_DISPLAY_NAME"
    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local -n values_ref="${schema_prefix}_VALUES"
    local -n choices_ref="${schema_prefix}_CHOICES"
    local var_name
    local raw_value
    local values=()
    local value

    for var_name in "${order_ref[@]}"; do
        raw_value="${values_ref[$var_name]-}"
        schema_split_raw_values "$raw_value" values

        if schema_var_is_required_current "$schema_prefix" "$var_name"; then
            if [[ -z "${values[0]}" ]]; then
                echo "${display_name_ref} variable '$var_name' is required for the current settings." >&2
                return 1
            fi
        fi

        for value in "${values[@]}"; do
            if ! schema_value_in_choices "$value" "${choices_ref[$var_name]}"; then
                echo "${display_name_ref} variable '$var_name' has invalid value '$value'. Allowed: ${choices_ref[$var_name]}" >&2
                return 1
            fi
        done
    done
}

# Validate shared recall settings after one schema sweep combination is loaded.
schema_validate_recall_config() {
    if [[ "${CALCULATE_RECALL:-False}" != "True" ]]; then
        return 0
    fi

    if [[ "${TASK^^}" != "QUERY" ]]; then
        echo "CALCULATE_RECALL=True is supported only for TASK=QUERY." >&2
        return 1
    fi

    if [[ -z "${GROUND_TRUTH_FILEPATH:-}" ]]; then
        echo "GROUND_TRUTH_FILEPATH is required when CALCULATE_RECALL=True." >&2
        return 1
    fi
}

# Print help output that describes every registered variable.
schema_print_registry_table() {
    local schema_prefix="$1"
    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local -n required_kind_ref="${schema_prefix}_REQUIRED_KIND"
    local -n required_if_ref="${schema_prefix}_REQUIRED_IF"
    local -n default_ref="${schema_prefix}_DEFAULT"
    local -n desc_ref="${schema_prefix}_DESC"
    local var_name
    local required_label
    local default_value

    printf "%-28s %-16s %-18s %-28s %s\n" "VARIABLE" "VALUE MODE" "REQUIRED" "DEFAULT" "CHOICES / NOTES"
    printf "%-28s %-16s %-18s %-28s %s\n" "--------" "----------" "--------" "-------" "---------------"

    for var_name in "${order_ref[@]}"; do
        required_label="${required_kind_ref[$var_name]}"
        if [[ -n "${required_if_ref[$var_name]}" ]]; then
            required_label="${required_label}:${required_if_ref[$var_name]}"
        fi

        default_value="${default_ref[$var_name]}"
        [[ -z "$default_value" ]] && default_value="<empty>"

        printf "%-28s %-16s %-18s %-28s %s\n" \
            "$var_name" \
            "single|sweep" \
            "$required_label" \
            "$default_value" \
            "${desc_ref[$var_name]}"
    done
}

# Print the resolved values after defaults and overrides have been applied.
schema_print_resolved_summary() {
    local schema_prefix="$1"
    local -n display_name_ref="${schema_prefix}_DISPLAY_NAME"
    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local -n values_ref="${schema_prefix}_VALUES"
    local var_name

    echo "Resolved ${display_name_ref} settings"
    printf '=%.0s' $(seq 1 $((10 + ${#display_name_ref} + 9)))
    echo
    for var_name in "${order_ref[@]}"; do
        printf "%-28s %s\n" "$var_name" "${values_ref[$var_name]-<unset>}"
    done
    echo
}

# Emit the cartesian product of all registered variable values.
#
# Each output line is a semicolon-delimited assignment list consumed by
# engine_load_combo.
schema_emit_combos_recursive() {
    local schema_prefix="$1"
    local index="$2"
    local combo_prefix="$3"
    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local -n values_ref="${schema_prefix}_VALUES"
    local var_name
    local values=()
    local value
    local next_prefix

    if (( index >= ${#order_ref[@]} )); then
        echo "${combo_prefix#;}"
        return 0
    fi

    var_name="${order_ref[$index]}"
    schema_split_raw_values "${values_ref[$var_name]-}" values

    for value in "${values[@]}"; do
        next_prefix="${combo_prefix};${var_name}=${value}"
        schema_emit_combos_recursive "$schema_prefix" $((index + 1)) "$next_prefix"
    done
}

# Load one emitted combo into shell globals.
#
# Variables may already exist as arrays from earlier override processing. Unset
# them before assignment so each combo value becomes a true scalar and stale
# sweep entries do not leak into run_config.env.
schema_load_combo_assignments() {
    local combo_line="$1"
    local assignment
    local key
    local value
    local assignments=()

    IFS=';' read -r -a assignments <<< "$combo_line"
    for assignment in "${assignments[@]}"; do
        [[ -n "$assignment" ]] || continue
        key="${assignment%%=*}"
        value="${assignment#*=}"
        unset "$key"
        printf -v "$key" '%s' "$value"
    done
}

# Emit the current scalar globals as KEY=value lines for run_config.env.
schema_emit_runtime_env() {
    local schema_prefix="$1"
    local -n order_ref="${schema_prefix}_VAR_ORDER"
    local var_name
    local raw_value

    for var_name in "${order_ref[@]}"; do
        raw_value="$(get_var_as_raw_string "$var_name" 2>/dev/null || true)"
        printf '%s=%s\n' "$var_name" "$raw_value"
    done
}
