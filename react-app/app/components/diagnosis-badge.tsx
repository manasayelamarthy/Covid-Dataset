interface DiagnosisBadgeProps {
  label: string
}

export function DiagnosisBadge({ label }: DiagnosisBadgeProps) {
  let colorClass = ""

  switch (label) {
    case "Covid":
      colorClass = "bg-red-100 text-red-800 border-red-200"
      break
    case "Viral Pneumonia":
      colorClass = "bg-amber-100 text-amber-800 border-amber-200"
      break
    case "Normal":
      colorClass = "bg-green-100 text-green-800 border-green-200"
      break
    default:
      colorClass = "bg-blue-100 text-blue-800 border-blue-200"
  }

  return <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${colorClass}`}>{label}</span>
}

