export const TechIcon = ({
  component,
  className = "size-10",
}: {
  component: React.ElementType;
  className?: string;
}) => {
  const Component = component;
  return (
    <>
      <Component className={`${className} fill-[url(#tech-icon-gradient)]`} />
      <svg className="size-0 absolute">
        <linearGradient id="tech-icon-gradient">
          <stop offset="0%" stopColor="#F3DFA7" />
          <stop offset="100%" stopColor="#C4A045" />
        </linearGradient>
      </svg>
    </>
  );
};
